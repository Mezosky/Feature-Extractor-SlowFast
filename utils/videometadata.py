"""script to get metadata from the videos"""

import os
import sys
import json
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_metadata(path_input: str, path_output: str) -> None:

    """
    This function get metadata info from a folder of videos. The
    output is a json file with the framerate, duration in seconds
    and total frames in the video.

    """

    print(
        "[Data] Loading metadata from the videos, this can took a lot of time..."
    )
    entries = os.listdir(path_input)
    json_file = {}

    for video_name in tqdm(entries):
        try:
            clip = VideoFileClip(os.path.join(path_input, video_name))
            json_file[video_name.split(".")[0]] = {
                "fps": clip.fps,
                "duration": clip.duration,
                "frames": int(clip.fps * clip.duration),
            }
        except:
            print(f"[Warning] Problem with {video_name}")
            continue

    with open(path_output, "w") as f:
        json.dump(json_file, f, indent=2)
        print("[Data] New json file was created")


if __name__ == "__main__":
    path_input = (
        sys.argv[1]
        if not sys.argv[1] is None
        else "/data/imeza/charades/Charades_v2_320_240"
    )
    path_output = (
        sys.argv[2]
        if not sys.argv[2] is None
        else "/data/imeza/charades/preprocessing/metadata_videos.json"
    )
    get_metadata(path_input, path_output)
