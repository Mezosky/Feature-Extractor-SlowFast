"""
Check the annotations created for anet with the different features

"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

def check_annotations(path1: str) -> None:

    # load video metadata (path1)
    with open(path1) as f:
        ann_videos = json.load(f)

    new_ann_videos, s_video, e_video = [], [], []
    
    # loop to check all the annotations
    for it, v in enumerate(ann_videos):
        if v['number_features'] < v['feature_end']:
            vi = v.copy()
            e_video.append(v)
            v['feature_end'] = v['number_features']
        if v['number_features'] < v['feature_start']:
            s_video.append(v)
        else:
            new_ann_videos.append(v)
    
    # Save the new Json File
    with open(path1, "w") as f:
        json.dump(new_ann_videos, f, indent=2)
        print("Json file was modified.")
    
if __name__ == "__main__":
    # annotations path
    path1 = str(sys.argv[1])

    check_annotations(path1)