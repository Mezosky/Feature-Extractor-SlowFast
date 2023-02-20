# HOWTO100M (on ABCI)

## Installation

First activate necessary modules to satisfy requirements.

```bash
module load gcc/11.2.0
module cuda/11.7/11.7.0
```

The proceed to install dependencies:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn
conda install av -c conda-forge
conda install -c conda-forge moviepy
conda install -c iopath iopath
conda install scipy
conda install ipdb -c conda-forge                                                              

pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
pip install psutil
pip install opencv-python
pip install tensorboard
pip install pytorchvideo
pip install moviepy
pip install decord

pip install 'git+https://github.com/facebookresearch/fairscale'

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2\
```

Now you're ready to run the installation:

```bash
git clone https://github.com/facebookresearch/slowfast
export $PYTHONPATH:/home/aad13288rp/slowfast/slowfast/
cd ~/slowfast
python setup.py build develop
```

If during installation you encounter the follwing problem:

```
Searching for PIL
Reading https://pypi.org/simple PIL/
No local packages or working download links found for PIL
```

Then run `pip install pillowcase` (if https://pypi.org/project/pillowcase/) and try again (source https://stackoverflow.com/questions/71524023/python-module-cropresize-wont-install-pil-dependency-doesnt-recognize-pillow)

## Comments

### C3D

Using 1,376 MB of memory ~80% GPU usage

```
02/20/2023 03:11:48 PM [Model Inference]] 1.- Video 7OvRe-fodcI Processed.
1443/1443 [00:37<00:00, 37.98it/s]

02/20/2023 03:12:26 PM [Model Inference] 2.- Processing 8KVPDvga1o8...
250/250 [00:04<00:00, 50.83it/s]

02/20/2023 03:12:31 PM [Model Inference] 3.- Processing 8o1Boy_o748...
468/468 [00:10<00:00, 46.01it/s]

02/20/2023 03:12:41 PM [Model Inference] 4.- Processing MSTpm3dNi1A...
366/366 [00:07<00:00, 48.20it/s]

02/20/2023 03:12:49 PM [Model Inference] 5.- Processing MwjnINh4yo0...
1121/1121 [00:22<00:00, 50.66it/s]

02/20/2023 03:13:11 PM [Model Inference] 6.- Processing PLv2O_eyTXY...
1209/1209 [00:26<00:00, 45.45it/s]

02/20/2023 03:13:38 PM [Model Inference] 7.- Processing gJOUv59Tkyc...
2411/2411 [00:58<00:00, 41.10it/s]

02/20/2023 03:14:37 PM [Model Inference] 8.- Processing lHX1AbnG7Zo...
1330/1330 [00:29<00:00, 45.59it/s]

02/20/2023 03:15:07 PM [Model Inference] 9.- Processing uDWskutOM0c...
1096/1096 [00:25<00:00, 42.54it/s]

02/20/2023 03:15:33 PM [Model Inference] 10.- Processing zUlhL1Ldo4g...
3342/3342 [01:21<00:00, 41.25it/s]

02/20/2023 03:16:54 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 5 minute(s) and 6 second(s)
```

To split into chunks:

```
python utils/splitdata.py /home/$USER/storage-kirt/datasets/howto100m-sample/videos /home/$USER 10
```

Which will place the csv's at the target directory (where the videos are). Then for each split you can run:

```
python run_net.py --cfg configs_files/howto100m-sample/C2D_8x8_R50.yaml --opts NUMBER_CSV 9
```

To run simultaneously, you can use:

```bash
# fish-shell
for num in (seq 10)                                                                            (howto) 
      python run_net.py --cfg configs_files/howto100m-sample/C2D_8x8_R50.yaml --opts NUMBER_CSV $num &
  end
```

Using ~7,000 MB of memory ~35% GPU usage

```
02/20/2023 04:44:45 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 36 minute(s) and 30 second(s)
02/20/2023 04:46:27 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 38 minute(s) and 12 second(s)
02/20/2023 04:47:53 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 39 minute(s) and 40 second(s)
02/20/2023 04:48:06 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 39 minute(s) and 52 second(s)
```