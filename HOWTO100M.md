# HOWTO100M (on ABCI)

## Installation

First activate necessary modules to satisfy requirements.

```bash
module load gcc/9.3.0 
module load cmake/3.22.3
module load cuda/11.3/11.3.1 
module load cudnn/8.3/8.3.3
```

The proceed to install dependencies:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install scikit-learn -y
conda install av -c conda-forge -y
conda install -c conda-forge moviepy -y
conda install -c iopath iopath -y
conda install scipy -y
conda install ipdb -c conda-forge -y

pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
pip install psutil
pip install opencv-python
pip install tensorboard
pip install pytorchvideo
pip install moviepy
pip install wrapt

# Below is installation of dependencies fro detectron2
# I recommend installing pytorchvideo from github as the pip is outdated
# pip install pytorchvideo
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
pip install 'git+https://github.com/facebookresearch/fairscale'
# now install detectron
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

If you want GPU support for decord, wow we need to install decord from source. Make sure the following is done on an instance with a GPU, not on the login node, otherwise some necessary CUDA libraries won't be available.
```bash
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build
cd build
CC=(which gcc) cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/apps/cuda/11.3.1/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release  
make
cd ../python
python3 setup.py install --use
```

Otherwise, simply run `pip install decord`.

Now you're ready to run the installation:

```bash
git clone https://github.com/facebookresearch/slowfast
export $PYTHONPATH:/home/aad13288rp/slowfast/slowfast/
cd ~/slowfast
python setup.py build develop
```

- If during installation you encounter the following problem:
```
Searching for PIL
Reading https://pypi.org/simple PIL/
No local packages or working download links found for PIL
```
Then run `pip install pillowcase` (if https://pypi.org/project/pillowcase/) and try again (source https://stackoverflow.com/questions/71524023/python-module-cropresize-wont-install-pil-dependency-doesnt-recognize-pillow)
If this does't work, consider commenting line declaring `PIL` as requirement in in `~/slowfast/setup.py` and install.

## Comments

### C2D

To split jobs into chunks you can use:

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
for num in (seq 10) 
      python run_net.py --cfg configs_files/howto100m-sample/C2D_8x8_R50.yaml --opts NUMBER_CSV $num &
  end
```

| Setting            | Num Videos | Time | Time/Video |
| ------------------ | ---------- | ---- | ---------- |
| No Parallelization | 100        |      |            |
|                    |            |      |            |
|                    |            |      |            |



- No parallelization, no cuda decord, Using 1,376 MB of memory ~80% GPU usage
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

No paralelization, cuda decord
```
2/27/2023 06:52:52 PM Processed 103 videos with the model ResNet,     it took a time of: 0 hour(s), 40 minute(s) and 19 second(s)
```


In 2 groups of 50:
```
02/22/2023 03:23:18 PM Processed 50 videos with the model ResNet,     it took a time of: 1 hour(s), 4 min ute(s) and 59 second(s) 

02/22/2023 03:19:10 PM Processed 50 videos with the model ResNet,     it took a time of: 1 hour(s), 0 minute(s) and 48` second(s)

```

- In 5 groups groups of 20 
```
02/21/2023 07:28:33 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 9 minute(s) and 5 second(s)
02/21/2023 07:27:57 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 7 minute(s) and 42 second(s)
02/21/2023 07:14:16 PM Processed 20 videos with the model ResNet,     it took a time of: 0 hour(s), 54 minute(s) and 34 second(s)
02/21/2023 07:28:33 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 9 minute(s) and 5 second(s)
02/21/2023 07:22:20 PM Processed 21 videos with the model ResNet,     it took a time of: 1 hour(s), 3 minute(s) and 7 second(s)
```


- In 10 groups of 10: Using ~10,000 MB of memory, ~60%  GPU usage
```
02/21/2023 10:25:21 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 42 minute(s) and 54 second(s)
02/21/2023 10:26:55 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 44 minute(s) and 27 second(s)
02/21/2023 10:31:48 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 49 minute(s) and 3 second(s)
02/21/2023 10:34:26 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 51 minute(s) and 14 second(s)
02/21/2023 10:26:25 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 43 minute(s) and 12 second(s)
02/21/2023 10:22:24 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 39 minute(s) and 3 second(s)
02/21/2023 10:16:04 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 31 minute(s) and 59 second(s)
---
02/21/2023 10:30:21 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 45 minute(s) and 47 second(s)
02/21/2023 10:34:48 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 50 minute(s) and 12 second(s)
```

- In 12 groups of ~8 each
```
02/22/2023 01:17:50 PM Processed 9 videos with the model ResNet,     it took a time of: 0 hour(s), 31 minute(s) and 48 second(s)
02/22/2023 01:20:04 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 33 minute(s) and 35 second(s)
02/22/2023 01:15:00 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 27 minute(s) and 26 second(s)
02/22/2023 01:07:39 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 21 minute(s) and 11 second(s)

02/22/2023 01:17:49 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 30 minute(s) and 3 second(s)
02/22/2023 01:21:06 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 32 minute(s) and 40 second(s)

```



- In 15 groups of ~7 each, goes OOM
- In 20 groups of 5 each, usinng 14,000 MB of RAM, ~35% GPU usage goes OOM

