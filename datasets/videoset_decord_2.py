import os
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



#logger = logging.get_logger(__name__)
# Set how default a torch tensor
decord.bridge.set_bridge('torch')

@DATASET_REGISTRY.register()
class VideoSetDecord2(torch.utils.data.Dataset):
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
            vr = VideoReader(path_to_vid, ctx=cpu(0))
            vr = vr.get_batch(range(0, len(vr), self.cfg.DATA.SAMPLING_RATE))

            self.in_fps = 30
            self.out_fps = 30
            self.step_size = int(self.in_fps / self.out_fps)

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
        
        frames = vr.clone()
        frames = frames.float()
        frames = frames / 255.0

        frames = tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            scl
        )
        relative_aspect = (
            asp
        )

        frames = spatial_sampling(
                    frames,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    #aspect_ratio=relative_aspect,
                    #scale=relative_scales,
                    #motion_shift=False,
                )

        return frames

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

        frame_seg = torch.zeros(
            (
                3,
                self.out_size,
                self.cfg.DATA.TEST_CROP_SIZE,
                self.cfg.DATA.TEST_CROP_SIZE,
            )
        ).float()

        start = int(index - self.step_size * self.out_size / 2)
        end = int(index + self.step_size * self.out_size / 2)
        max_ind = self.__len__() - 1

        for out_ind, ind in enumerate(range(start, end, self.step_size)):
            if ind < 0 or ind > max_ind:
                continue
            else:
                frame_seg[:, out_ind, :, :] = self.frames[:, ind, :, :]

        # create the pathways
        frame_list = pack_pathway_output(self.cfg, frame_seg)

        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        # return self.video_container.streams.video[0].frames
        return self.frames.shape[1]