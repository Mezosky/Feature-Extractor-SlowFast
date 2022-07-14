from decord import VideoReader
from decord import cpu, gpu
import decord
import torch

decord.bridge.set_bridge('torch')

path = '/data/imeza/charades/Charades_v1_320_240/1YC83.mp4'
vr = VideoReader(path, ctx=cpu(0))

# Create a crop function
def _crop_image(image, crop_size):
    w, h, _ = image.shape
    cw, ch = crop_size, crop_size
    return image[w//2 - cw//2:w//2 + cw//2, 
                 h//2 - ch//2:h//2 + ch//2, 
                 ...
                ]

SAMPLING_RATE = 3
vr = vr.get_batch(range(0,len(vr), SAMPLING_RATE))

crop = 224

frames = torch.zeros(vr.shape[0], crop, crop, 3)
for i in range(len(vr)):
    frames[i] = _crop_image(vr[i], crop)

print(frames.shape)

