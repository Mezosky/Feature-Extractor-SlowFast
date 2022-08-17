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

# How to run

## I3D

```python
python run_net.py --cfg "./configs/I3D_8x8_R50_ALL.yaml"
```

```python
python multiple_runs.py --cfg "./configs/I3D_8x8_R50.yaml"
```


## SlowFast

```python
python run_net.py --cfg "./configs/SLOWFAST_8x8_R50_ALL.yaml"
```

## X3D

```python
python run_net.py --cfg "./configs/X3D_M.yaml"
```

## MVit

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