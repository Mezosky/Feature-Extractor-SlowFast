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
class VideoSetDecord5(torch.utils.data.Dataset):
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

        self.out_size = cfg.DATA.NUM_FRAMES

        self.vid_path = vid_path
        self.vid_id = vid_id
        self.path_to_vid = (
            os.path.join(self.vid_path, self.vid_id) + self.cfg.DATA.VID_EXT
        )

        if self.cfg.LOAD_SHORT_VIDEOS:
            self.frames = self._get_frames_short_video()
        else:
            self._get_frames_long_video()

    def _check_video(self):

        assert os.path.exists(self.path_to_vid), "{} file not found".format(self.path_to_vid)

        try:
            # set the step size, the input and output
            # Load frames
            frames = VideoReader(self.path_to_vid, ctx=cpu(0))
            self.step_size = 1

        except Exception as e:
            logger.info(
                f"Failed to load video from {self.path_to_vid} with error {e}"
            )

        return frames

    def _transform_frames(self, frames):
        """
        Performs a treatment on the frames, transforming them to the 
        standard of the models used in pySlowFast.
        
        Args:
            frames (Tensor)
        Return
            frames (Tensor)
        """

        # Resize the frames
        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        
        # Normalize the frames
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

        return frames


    def _get_frames_short_video(self):
        """
        Extract frames from the video container
        Returns:
            frames(tensor or list): A tensor of extracted frames from a video or a list of images to be processed
        """

        frames = self._check_video()        
        frames = frames.get_batch(range(0, len(frames), self.cfg.DATA.SAMPLING_RATE))
        frames = self._transform_frames(frames)

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

    def _get_frames_long_video(self):
        self.q_frames = len(self._check_video())
        print(self.q_frames)
        # we define a tuple where we'll store the index and bath of frames
        # The first postion is a list where we'll store the indexes contained
        # in the bath of frames.
        self.tuple_idx_frame = ([], None)
        # Define the large of the batch of frames
        self.frames_batch = self.q_frames if self.q_frames < 96 else 96
        # you get a tuple containing the indexes and frames
        self.idx_video_tuples = self._gen_range_idx_frame(0, self.q_frames, self.frames_batch, self.out_size)

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

        if self.cfg.LOAD_SHORT_VIDEOS:
            frame_list = pack_pathway_output(self.cfg, self.frames[index])
            return frame_list
        
        ############## long videos ###########
        if index in self.tuple_idx_frame[0]:
            # Get the batch to load
            i = index - self.tuple_idx_frame[0].min()
            s, e = int(i*self.out_size), int((i+1)*self.out_size)
            output_frames = self.tuple_idx_frame[1][:, s:e, ...]
        elif index not in self.tuple_idx_frame[0]:
            try:
                # get tuples
                idx_tuple = self._get_tuple_idx(index, self.idx_video_tuples)
                index_list = np.array(self.idx_video_tuples[idx_tuple][0])

                # load video
                start_end_tuple = self.idx_video_tuples[idx_tuple][1]
                vr = VideoReader(self.path_to_vid, ctx=cpu(0))
                frames = vr.get_batch(range(start_end_tuple[0], start_end_tuple[1], 1))
                frames = self._transform_frames(frames) # prueba

                # save in a "cache" the loaded segment
                self.tuple_idx_frame = (index_list, frames)

                # get the output frame
                i = index - np.array(index_list).min()
                s, e = int(i*self.out_size), int((i+1)*self.out_size)
                output_frames = frames[:, s:e, ...]
                #output_frames = frames[s:e, ...]
                #output_frames = self._transform_frames(output_frames)
            except AssertionError as error:
                print(error)

        return pack_pathway_output(self.cfg, output_frames)

    def __len__(self):
        """
        Returns:
            (int): the number of indixes in the video.
        """

        if self.cfg.LOAD_SHORT_VIDEOS:
            return len(self.frames)
        else:
            return self.idx_video_tuples[-1][0][-1]

    @staticmethod
    def _gen_range_idx_frame(a, b, c, d):
        """
        Create a list with the indices of the video and
        the frames that contains these indices.
        Args:
            a (int): First frame  
            b (int): Last frame
            c (int): batch of frames
            d (int): sampling frame rate
        """

        idx_start, idx_end = 0, int(c/d)
        list_out =[]
        for i in range(a, b, c):
            if (i+c) < b:
                idx_frame_tuple = (range(idx_start, idx_end), (i, i+c))
                idx_start, idx_end = idx_end, idx_end + int(c/d)
                list_out.append(idx_frame_tuple)
        # Check if we have some extra frames to save
        if b - i >= d:
            idx_frame_tuple = (range(idx_start, idx_start + int((b - i)/d)), (i, i+d*int((b - i)/d)))
            list_out.append(idx_frame_tuple)
        
        return list_out         

    @staticmethod
    def _get_tuple_idx(idx, q_tuple):
        out = 0
        for t in q_tuple:
            if idx in t[0]:
                return out
            out += 1
        return out