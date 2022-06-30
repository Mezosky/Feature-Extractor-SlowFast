# Features Extractor SlowFast

The next code is focus to get features from the SlowFast library of Facebook.

Note: Sometimes `moviepy` can give some issues to run the code, in that case please try:
it
```cmd
pip uninstall moviepy
pip install moviepy
```
# Installation

To install and execute the current code you must install the [SlowFast library](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md).

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
