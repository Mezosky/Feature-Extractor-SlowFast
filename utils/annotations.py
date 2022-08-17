"""
Create annotations with charades for the different features

how to run:
python utils/annotations.py '/data/imeza/charades/preprocessing/metadata_videos.json' "/data/imeza/charades/Charades_Features_v2/MViT" '/home/imeza/TMLGA/preprocessing/charades-sta'
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

def get_annotations(path1, 
                    path2, 
                    path3):

    # train file created by Cristian
    with open('/data/imeza/charades/annotations/charades_sta_train_tokens.json') as f:
        json_train = json.load(f)

    # test file created by Cristian
    with open('/data/imeza/charades/annotations/charades_sta_test_tokens.json') as f:
        json_test = json.load(f)
    # Create a list with the annotations files 
    list_annotations_files = [json_train, json_test]

    # load video metadata (path1)
    with open(path1) as f:
        metadata_video = json.load(f)

    model_name = path2.split("/")[-1]

    new_json = []
    for it, ann_file in enumerate(list_annotations_files):
        for e_file in tqdm(ann_file):
            fps = metadata_video[e_file['video']]['fps']
            frames = int(metadata_video[e_file['video']]['frames'])
            
            fpath = os.path.join(path2, f"{e_file['video']}.npy")
            n_features = np.load(fpath).shape[0]
            
            e_file['frame_start'] = e_file['time_start'] * fps 
            e_file['frame_end']   = e_file['time_end'] * fps
            e_file['feature_start'] = e_file['frame_start'] / (frames/n_features)
            e_file['feature_end'] =  e_file['frame_end'] / (frames/n_features)
            e_file['number_features'] = n_features
            e_file['number_frames'] = frames
            e_file['fps'] = fps
            e_file['preprocessing'] = 'imeza_v0'
            e_file['model'] = model_name
            
            new_json.append(e_file)

        if it == 0:
            new_json_name = f"charades_sta_train_tokens_{model_name}.json"
        else:
            new_json_name = f"charades_sta_test_tokens_{model_name}.json"

        path_output = os.path.join(path3, new_json_name)

        with open(path_output, 'w') as f:
            json.dump(new_json, f, indent=2)
            print("New json file was created")

if __name__ == "__main__":
    path1 = str(sys.argv[1])
    path2 = str(sys.argv[2])
    path3 = str(sys.argv[3])
    
    get_annotations(path1, path2, path3)
