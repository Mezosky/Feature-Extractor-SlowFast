"""function to get metadata from the videos"""

import os
import sys
import json
import torch
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def main():

    print("Loading metadata from the videos, this can took a lot of time...")
    path    = "/data/imeza/charades/Charades_v2_320_240"
    entries = os.listdir(path)

    json_file = {}
    for video_name in tqdm(entries):
        try:
            clip = VideoFileClip(os.path.join(path, video_name))
            json_file[video_name.split(".")[0]] = \
            {"fps":clip.fps, "duration":clip.duration, "frames":int(clip.fps * clip.duration)}
        except:
            print(f"Problem with {video_name}")
            continue

    with open('/data/imeza/charades/preprocessing/metadata_videos.json', 'w') as f:
        json.dump(json_file, f, indent=2)
        print("New json file was created")

if __name__ == "__main__":
    main()