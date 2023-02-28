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


### On A100 nodes

If you encouter the following error:
```bash
symbol lookup error: ssh: undefined symbol: EVP_KDF_ctrl, version OPENSSL_1_1_1b
```
That means that potentially SSL library conflicts on CentOS 8 #10241, as explained here https://github.com/conda/conda/issues/10241. To solve run `export LD_PRELOAD=/usr/lib64/libcrypto.so`

Finally
```bash
export DECORD_EOF_RETRY_MAX=20480
```


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



-V100 CPU decord, no parallelization: Using 1,376 MB of memory ~80% GPU usage
```
02/20/2023 03:16:54 PM Processed 10 videos with the model ResNet,     it took a time of: 0 hour(s), 5 minute(s) and 6 second(s)
```

V100 CUDA decord, no paralelization:
```
2/27/2023 06:52:52 PM Processed 103 videos with the model ResNet,     it took a time of: 0 hour(s), 40 minute(s) and 19 second(s)
```

V100 CPU decord, in 2 groups of 50:
```
02/22/2023 03:23:18 PM Processed 50 videos with the model ResNet,     it took a time of: 1 hour(s), 4 min ute(s) and 59 second(s) 
02/22/2023 03:19:10 PM Processed 50 videos with the model ResNet,     it took a time of: 1 hour(s), 0 minute(s) and 48` second(s)
```

- V100 CPU decord, in 5 groups groups of 20:
```
02/21/2023 07:28:33 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 9 minute(s) and 5 second(s)
02/21/2023 07:27:57 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 7 minute(s) and 42 second(s)
02/21/2023 07:14:16 PM Processed 20 videos with the model ResNet,     it took a time of: 0 hour(s), 54 minute(s) and 34 second(s)
02/21/2023 07:28:33 PM Processed 20 videos with the model ResNet,     it took a time of: 1 hour(s), 9 minute(s) and 5 second(s)
02/21/2023 07:22:20 PM Processed 21 videos with the model ResNet,     it took a time of: 1 hour(s), 3 minute(s) and 7 second(s)
```


- V100 CPU decord, in 10 groups of 10: Using ~10,000 MB of memory, ~60%  GPU usage
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

- V100 CPU decord in 12 groups of ~8 each: 
```
02/22/2023 01:17:50 PM Processed 9 videos with the model ResNet,     it took a time of: 0 hour(s), 31 minute(s) and 48 second(s)
02/22/2023 01:20:04 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 33 minute(s) and 35 second(s)
02/22/2023 01:15:00 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 27 minute(s) and 26 second(s)
02/22/2023 01:07:39 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 21 minute(s) and 11 second(s)

02/22/2023 01:17:49 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 30 minute(s) and 3 second(s)
02/22/2023 01:21:06 PM Processed 8 videos with the model ResNet,     it took a time of: 0 hour(s), 32 minute(s) and 40 second(s)
```


A100 CUDA decord, no parallelization, Mmo: 
```
02/28/2023 05:00:16 PM Processed 103 videos with the model ResNet,     it took a time of: 0 hour(s), 23 minute(s) and 45 second(s)
```

A100 CUDA decord, in 2 groups of 50:
```
02/28/2023 05:29:57 PM Processed 50 videos with the model ResNet,     it took a time of: 0 hour(s), 21 minute(s) and 47 seconds
02/28/2023 05:28:28 PM Processed 50 videos with the model ResNet,     it took a time of: 0 hour(s), 20 minute(s) and 11 seconds
``` 

