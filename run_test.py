import torch
import argparse

import logging
import sys
import json
import os
import time

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args
from configs.custom_config import load_config

from get_features import test

import ipdb
import GPUtil


def benchmark(input_path: str) -> None:
    # Initializing the parse setting with the facebook framework
    args = parse_args()
    # Load config files names
    configs_files = os.listdir(input_path)

    # initializing the logger
    logging.basicConfig(
        filename="benchmark_executions.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
    )
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)
    log = logging.getLogger(__name__)

    # logging the initialization
    log.info("Initializing the Logger")

    # dictionary to save execution data
    json_dict = {}
    json_gpu_dict = {}

    for config_file in configs_files:
        print(config_file)

        # Start of run time
        start_time = time.time()
        name = str(config_file.split(".")[0])

        # Load a config yaml
        config_path = os.path.join(input_path, config_file)
        cfg = load_config(args, config_path)

        # Select GPU
        torch.cuda.set_device(0)
        log.info("[GPU]: Device", torch.cuda.current_device())

        # Check if the cfg file is for test
        if cfg.TEST.ENABLE:
            try:
                launch_job(cfg=cfg, init_method=args.init_method, func=test)
            except:
                log.info(f"Error with the model {name}.")
        else:
            raise Exception(
                "This function can only get features, classification \
            is not implemented. Please change TEST.ENABLE to True in the .yaml file."
            )

        # Time of completion of execution
        final_time = time.time() - start_time
        log.info(f"[Benchmark] The model {name} took {final_time} [s]")

        # Clear memory
        torch.cuda.empty_cache()

        # Save data in the dictionary
        json_dict[config_file] = final_time
        json_gpu_dict[config_file] = GPUtil.getGPUs()[0].memoryUsed

    # Save execution time in json
    log.info("[Benchmark-Data] Saving data...")
    if not os.path.exists("./test/feat_output/"):
        os.mkdir("./test/feat_output/")
    with open("./test/feat_output/execution_time.json", "w") as outfile:
        json.dump(json_dict, outfile)
    with open("./test/feat_output/gpu_info.json", "w") as outfile:
        json.dump(json_gpu_dict, outfile)


if __name__ == "__main__":

    benchmark("./configs_files/test/")
