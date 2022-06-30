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

- [ ] Change to ffmpeg the resize function of the videos.
- [ ] Generate a function to get the features in paralell.
- [ ] Analyze how the size of the input influences the feature extractor.

# How to run

## I3D

```python
python run_net.py --cfg "./configs/I3D_8x8_R50.yaml"
```

## SlowFast

```python
python run_net.py --cfg "./configs/SLOWFAST_8x8_R50.yaml"
```

## MVit

```python
python run_net.py --cfg "./configs/MVIT_B_16x4_CONV.yaml"
```

```python
python run_net.py --cfg "./configs/MVIT_B_32x3_CONV.yaml"
```
