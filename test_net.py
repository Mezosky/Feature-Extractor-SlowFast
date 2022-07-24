"""
Extract features for videos using pre-trained arquitectures

ToDo:
 - Clean this Code and generate a Logging.
"""

import numpy as np
import pandas as pd
import torch
import os
import time
from tqdm import tqdm
import av
from moviepy.video.io.VideoFileClip import VideoFileClip

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from models import build_model
from datasets import VideoSet
from datasets import VideoSetDecord
from datasets import VideoSetDecord2

#logger = logging.get_logger(__name__)


def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds

def create_csv(path, max_files='all'):
    assert (
        type(max_files) is not int or max_files != 'all'
    ), "You must enter a int from the 1 to the N"

    if max_files == 'all':
        entries = os.listdir(path)
    else:
        entries = os.listdir(path)
        entries = np.array_split(entries, max_files)
    
    entries = os.listdir(path)
    entries = [v.split(".")[0] for v in entries]
    
    path_csv = path + '/vid_list.csv'
    if os.path.exists(path_csv):
        os.remove(path_csv)

    df = pd.DataFrame(entries)

    if max_files == 'all':
        df.to_csv(path_csv,index=False, header=False)
    else:
        indices = np.array_split(df.index, max_files)
        for i in range(max_files):
            df_i = df.loc[[indices[i]]]
            df.to_csv(path + f'/vid_list_{i}.csv', index=False, header=False)


@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    feat_arr = None

    for inputs in tqdm(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Perform the forward pass.
        preds, feat = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])

        feat = feat.cpu().numpy()

        if feat_arr is None:
            feat_arr = feat
        else:
            feat_arr = np.concatenate((feat_arr, feat), axis=0)

    return feat_arr


def test(cfg):
    """
    Perform multi-view testing/feature extraction on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Create ouput folder.
    output_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"The directory {cfg.MODEL.MODEL_NAME} was created!")
    
    # Setup logging format.
    #logging.setup_logging(output_path)

    # Print config.
    #logger.info("Test with config:")
    #logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)
    
    vid_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
    create_csv(cfg.DATA.PATH_TO_DATA_DIR)

    videos_list_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "vid_list.csv")

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        videos = sorted([x.strip() for x in f.readlines() if len(x.strip()) > 0])
    print("Done")
    print("----------------------------------------------------------")

    
    rejected_vids = []

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")

    start_time = time.time()
    for vid_no, vid in enumerate(videos):
        # Create video testing loaders.
        path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
        vid_id = os.path.split(vid)[1]

        try:
            _ = VideoFileClip(
                os.path.join(path_to_vid, vid_id) + ".mp4",
                audio=False,
                fps_source="fps",
            )
        except Exception as e:
            print("{}. {} cannot be read with error {}".format(vid_no + 1, vid, e))
            print("----------------------------------------------------------")
            rejected_vids.append(vid)
            continue

        out_path = os.path.join(output_path, os.path.split(vid)[0])
        out_file = vid_id.split(".")[0] + ".npy"
        if os.path.exists(os.path.join(out_path, out_file)):
            print("{}. {} already exists".format(vid_no + 1, out_file))
            print("----------------------------------------------------------")
            continue

        print("{}. Processing {}...".format(vid_no + 1, vid))

        dataset = VideoSetDecord2(
            cfg, path_to_vid, vid_id, #read_vid_file=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            sampler=None,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
        )

        # Perform multi-view test on the entire dataset.
        feat_arr = perform_inference(test_loader, model, cfg)

        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, out_file), feat_arr)
        del dataset
        del test_loader
        print("Done.")
        print("----------------------------------------------------------")

    
    print("Rejected Videos: {}".format(rejected_vids))

    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    # add here a loging
    print(
        f"Time taken: {hours} hour(s), {minutes} minute(s) and {seconds} second(s)"
    )
    print("----------------------------------------------------------")