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
from slowfast.datasets import DATASET_REGISTRY
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)
# Set how default a torch tensor
decord.bridge.set_bridge('torch')

@DATASET_REGISTRY.register()
class VideoSetDecord(torch.utils.data.Dataset):
    """
    Construct the untrimmed video loader, then sample
    segments from the videos. The videos are segmented by centering
    each frame as per the output size i.e. cfg.DATA.NUM_FRAMES.
    """

    def __init__(self, cfg, vid_path, vid_id, read_vid_file=False):
        """
        Construct the video loader for a given video.
        Args:
            cfg (CfgNode): configs.
            vid_path (string): path to the video
            vid_id (string): video name
            read_vid_file (bool): flag to turn on/off reading video files.
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id
        self.read_vid_file = read_vid_file

        self.out_size = cfg.DATA.NUM_FRAMES

        if isinstance(self.cfg.DATA.SAMPLE_SIZE, int):
            self.sample_width, self.sample_height = [self.cfg.DATA.SAMPLE_SIZE, 
                                                     self.cfg.DATA.SAMPLE_SIZE]
        elif isinstance(self.cfg.DATA.SAMPLE_SIZE, list):
            self.sample_width, self.sample_height = self.cfg.DATA.SAMPLE_SIZE
        else:
            raise Exception(
                "Error: Frame sampling size type must be an int"
            )

        self.frames = self._get_frames()

    def _crop_image(image, crop_size):
        """
        Cut an image given a crop size.
        Input:
            image(tensor): Tensor of a image/frame
            crop_size: Crop size
        Return:
            frame(tensor): A tensor with the image cut off

        """
        w, h, _ = image.shape
        cw, ch = crop_size, crop_size
        
        return image[w//2 - cw//2:w//2 + cw//2, 
                    h//2 - ch//2:h//2 + ch//2, 
                    ...
                    ]

    def _get_frames(self):
        """
        Extract frames from the video container
        Returns:
            frames(tensor or list): A tensor of extracted frames from a video or a list of images to be processed
        """
        if self.read_vid_file:
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
                    "Failed to load video from {} with error {}".format(path_to_vid, e)
                )

            frames = torch.zeros(vr.shape[0], self.cfg.DATA.TEST_CROP_SIZE, self.cfg.DATA.TEST_CROP_SIZE, 3)
            for i in range(len(vr)):
                height, width, _ = vr[i].shape
                if height != self.sample_height and width != self.sample_width:
                    frames[i] = cv2.resize(
                        vr[i],
                        (self.sample_width, self.sample_height),
                        interpolation=cv2.INTER_LINEAR
                        )
                    frames[i] = self._crop_image(frame[i], self.cfg.DATA.TEST_CROP_SIZE)

                else:
                    frames[i] = self._crop_image(vr[i], self.cfg.DATA.TEST_CROP_SIZE)
                    w, h, _ = vr[i].shape
                    cw, ch = self.cfg.DATA.TEST_CROP_SIZE, self.cfg.DATA.TEST_CROP_SIZE
                    frames[i] = vr[i][w//2 - cw//2:w//2 + cw//2, h//2 - ch//2:h//2 + ch//2, ...]
            frames = self._pre_process_frame(frames)

            return frames

        else:
            raise Exception(
                "Error: You have not entered a video in MP4 format to load."
            )
            

    def _pre_process_frame(self, arr):
        """
        Pre process an array
        Args:
            arr (ndarray): an array of frames or a ndarray of an image
                of shape T x H x W x C or W x H x C respectively
        Returns:
            arr (tensor): a normalized torch tensor of shape C x T x H x W 
                or C x W x H respectively
        """
        if type(arr) == np.ndarray:
            arr = torch.from_numpy(arr).float()

        # Normalize the values
        arr = arr / 255.0
        arr = arr - torch.tensor(self.cfg.DATA.MEAN)
        arr = arr / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        if len(arr.shape) == 4:
            arr = arr.permute(3, 0, 1, 2)
        elif len(arr.shape) == 3:
            arr = arr.permute(2, 0, 1)

        return arr

    def _read_img_file(self, path, file):
        """
        Read an image file
        Args:
            path (str): path to the image file
            file (str): name of image file
        Returns:
            img (tensor): a normalized torch tensor
        """
        img = cv2.imread(os.path.join(path, file))

        if len(img.shape) != 3:
            raise Exception(
                "Incorrect image format. Image needs to be read in RGB format."
            )

        img = cv2.resize(
            img,
            (self.sample_width, self.sample_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img = self._pre_process_frame(img)
        return img

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
                if self.read_vid_file:
                    frame_seg[:, out_ind, :, :] = self.frames[:, ind, :, :]
                else:
                    frame_seg[:, out_ind, :, :] = self._read_img_file(
                        os.path.join(self.vid_path, self.vid_id), self.frames[ind]
                    )

        # create the pathways
        frame_list = pack_pathway_output(self.cfg, frame_seg)

        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        # return self.video_container.streams.video[0].frames
        if self.read_vid_file:
            return self.frames.shape[1]
        else:
            return len(self.frames)