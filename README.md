# 🎥 Features Extractor SlowFast 🎥

The following code has as main objective to obtain features using the PySlowFast meta library. The code provided here is focused only on obtaining features, for this reason it is not possible to obtain inference in a native way in its execution.

Please if you want to use the code read the "installation" and "How to use" section. For the execution of the script it is necessary to set/define in the configuration file some relevant inputs for each model.

# Installation

To install and run the current code, you must install the [SlowFast library](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md). In other hand, you must install:

```
pip install scipy
pip install moviepy
```

Note: Sometimes `moviepy` may give some problems to execute the code, in that case please try this:

```cmd
pip uninstall moviepy
pip install moviepy
```

# How to run

To execute the code see the next instructions, here you will find the execution script for each supported model (see the supported models here).

# Checkpoints

To load weights for Resnet, SlowFast and MViT models, use the following [weights](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md).

# Supported Models