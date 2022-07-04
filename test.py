import os
import time
import nvidia_smi
from joblib import Parallel, delayed

def split_folder(path_in):
    pass    

def comprobate_run(parallel_job):

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    total_mem = round(info.total/(1024**3), 3)
    free_mem = round(info.free/(1024**3), 3)

    try:
        exec(os.system("python run_net.py --cfg './configs/MVIT_B_32x3_CONV_ALL.yaml'"))
        print(f"Running {parallel_job} process")
        print(f"Total memory: {total_mem}")
        print(f"Free memory: {free_mem}")
    except:
        print("#"+"-"*48+"#")
        print("Not enough free GPU memory.")   
        print("#"+"-"*48+"#")
 
if __name__ == "__main__":
    Parallel(n_jobs=-1)(delayed(comprobate_run)(it) for it in range(3))