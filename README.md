# Features Extractor SlowFast

The following code is focused on getting features from Facebook's SlowFast library.

Note: Sometimes `moviepy` may give some problems to execute the code, in that case please try this:

```cmd
pip uninstall moviepy
pip install moviepy
```
# Installation

To install and run the actual code, you must install the [SlowFast library](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md).In other hand, you must install:

```
pip install scipy
pip install moviepy
```

# Checkpoints

To load weights for Resnet, SlowFast and MViT models, use the following [weights](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md).

# TO-DO

- [X] Change to ffmpeg the resize function of the videos.
- [X] Generate a function to get the features in paralell.
- [X] Add the creation of csv in test.py
- [X] Add args to test.py and add an extra run_net.py (solved with the yaml)
- [ ] Analyze how the size of the input influences the feature extractor.
- [X] Modify the run_net.py to process different video_list files. (`multiples_run.py`)
    - [X] Create get_features function to process the features.
    - [X] Modify get_features to process multiples vid_list_i.csv
- [X] Check the code and change more things in `get_features`
- [X] Add Decord to the code
- [X] Add a crop function to cut the image from the center
- [ ] Compare the dataloaders
- [x] Create a dataloader with the charades and kinetics class from pyslowfast
- [x] Add video format selector
- [x] Add X3D encoder
    - [X] Add X3D head
    - [X] Add X3D forward
    - [X] Modify head in build.py
    - [X] Add yaml
- [ ] X3D has a bug with the framerate, I have to check why is generating a big sampling of frames
- [ ] Check number of frames in videos
    - [ ] Add a fps reader to the inputs videos
- [ ] Check the output feat for each model.
- [ ] Create a new csv in each run. each run have to detect if we have the file npy processed.

# How to run

## I3D

```python
python run_net.py --cfg "./configs/I3D_8x8_R50.yaml"
```

```python
python multiple_runs.py --cfg "./configs/I3D_8x8_R50.yaml"
```


## SlowFast

```python
python run_net.py --cfg "./configs/SLOWFAST_8x8_R50.yaml"
```

## X3D

```python
python run_net.py --cfg "./configs/X3D_M.yaml"
```

## MVit

```python
python run_net.py --cfg "./configs/MVIT_B_16x4_CONV.yaml"
```

```python
python run_net.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

```python
python multiple_runs.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

---

charades.py

```python
python charades_test.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

# Questions

- [X] What is the CROP size and JITTER size?.
- [X] is 320:240 the size of video?. This is because I saw in the yaml of the slowfast different text_crop_size.
- [X] We may change the frames per second on the videos?

# Blocking

- [ ] I don't know what is `_spatial_temporal_idx`, this variable is associated with the framerate of the video but at the same time is related with the label and video's index.

