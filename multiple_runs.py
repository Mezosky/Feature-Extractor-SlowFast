import os
import time
import torch
import nvidia_smi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args
from configs.custom_config import load_config

from get_features import test

def create_csv(path, output_path,max_files='all'):
    assert (
        type(max_files) is not int or max_files != 'all'
    ), "You must enter a int from the 1 to the N"
    
    for f in os.listdir(path):
        if "csv" in f:
            os.remove(path+"/"+f) 

    if os.path.exists(output_path):
        proc_v = [v.split(".")[0] for v in os.listdir(output_path)]
        if len(proc_v) > 0:
            print(f"Already {len(proc_v)} files have been processed")
        entries = os.listdir(path)
        entries = [v.split(".")[0] for v in entries if v.split(".")[0] not in proc_v]
    else:
        entries = os.listdir(path)
        entries = [v.split(".")[0] for v in entries]

    df = pd.DataFrame(entries)

    if max_files == 'all':
        path_csv = path + '/videos_list.csv'
        if os.path.exists(path_csv):
            os.remove(path_csv)
        df.to_csv(path_csv,index=False, header=False)

    else:
        indices = np.array_split(df.index, max_files)
        for i in range(max_files):
            path_csv = path + f'/videos_list_{i+1}.csv'
            if os.path.exists(path_csv):
                os.remove(path_csv)

            data_csv = df.loc[indices[i]]
            data_csv.to_csv(path_csv, index=False, header=False)

def comprobate_run(cfg, args, parallel_job):

    torch.cuda.set_device(0)
    print('Device GPU: ', torch.cuda.current_device())

    cfg.ITERATION = parallel_job + 1

    nvidia_smi.nvmlInit()
    # I don't know why the index has another number
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    total_mem = round(info.total/(1024**3), 3)
    free_mem = round(info.free/(1024**3), 3)

    # Search how to set a not manual treshold
    if free_mem > 4:
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)
        print(f"Running process {cfg.ITERATION}")
        print(f"Total memory: {total_mem}")
        print(f"Free memory: {free_mem}")
    else:
        print("Not enough free GPU memory.")   
 
if __name__ == "__main__":
    
    args = parse_args()

    for config_path in args.cfg_files:
        cfg = load_config(args, config_path)
        output_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)
        create_csv(cfg.DATA.PATH_TO_DATA_DIR, output_path, max_files=cfg.MULTIPLE_PROCESS)
        # change the number inside the range with a parameter from cfg file
        Parallel(n_jobs=cfg.MULTIPLE_PROCESS)(delayed(comprobate_run)(cfg, args, it) for it in range(cfg.MULTIPLE_PROCESS))