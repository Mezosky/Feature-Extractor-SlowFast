import os
import math
import random
from io import BytesIO
import torch
import torch.utils.data
import numpy as np
from parse import parse
import av
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from decord import VideoReader
from decord import cpu, gpu
import decord

import slowfast.datasets.decoder as decoder
import slowfast.datasets.transform as transform
import slowfast.datasets.video_container as container
from slowfast.datasets.utils import pack_pathway_output
from slowfast.datasets.utils import spatial_sampling
from slowfast.datasets.utils import tensor_normalize
from slowfast.datasets import DATASET_REGISTRY
import slowfast.utils.logging as logging



logger = logging.get_logger(__name__)
# Set how default a torch tensor
decord.bridge.set_bridge('torch')

@DATASET_REGISTRY.register()
class VideoSetDecord4(torch.utils.data.Dataset):
    """
    Construct the untrimmed video loader, then sample
    segments from the videos. The videos are segmented by centering
    each frame as per the output size i.e. cfg.DATA.NUM_FRAMES.
    """

    def __init__(self, cfg, vid_path, vid_id):
        """
        Construct the video loader for a given video.
        Args:
            cfg (CfgNode): configs.
            vid_path (string): path to the video
            vid_id (string): video name
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id

        self.out_size = cfg.DATA.NUM_FRAMES
        self.frames = self._get_frames()

    def _get_frames(self):
        """
        Extract frames from the video container
        Returns:
            frames(tensor or list): A tensor of extracted frames from a video or a list of images to be processed
        """
        path_to_vid = (
            os.path.join(self.vid_path, self.vid_id) + ".mp4"
        )
        assert os.path.exists(path_to_vid), "{} file not found".format(path_to_vid)


        try:
            # set the step size, the input and output
            # Load frames
            frames = VideoReader(path_to_vid, ctx=cpu(0))
            frames = frames.get_batch(range(0, len(frames), self.cfg.DATA.SAMPLING_RATE))
            self.step_size = 1

        except Exception as e:
            logger.info(
                f"Failed to load video from {path_to_vid} with error {e}"
            )
        
        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        
        frames = tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        frames = spatial_sampling(
                    frames,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )

        # generamos una lista con los valores agrupados
        step_size = self.cfg.DATA.NUM_FRAMES
        iterations = math.ceil(frames.shape[1]/step_size)

        CZ, _, HZ, WZ = frames.shape
   
        frames_list = []
        for it in range(iterations):
            
            start = it*step_size
            end   = (it+1)*step_size

            frames_batch = frames[:, start:end, :, :]
            q_frames = frames_batch.shape[1]
            if q_frames < step_size:
                frames_zeros = torch.zeros(CZ, step_size, HZ, WZ)
                frames_zeros[:, :q_frames, :, :] = frames_batch[:, :q_frames, :, :]
                frames_batch = frames_zeros.clone()
            frames_list.append(frames_batch)

        return frames_list

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        frame_list = pack_pathway_output(self.cfg, self.frames[index])
        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        # return self.video_container.streams.video[0].frames
        return len(self.frames)