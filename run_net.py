import torch
import logging
import sys

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args
from configs.custom_config import load_config

from get_features import test

def main():
    """
    Main function to process the videos.
    """
    args = parse_args()

    for path_to_config in args.cfg_files:
        # Initialiaze the logger
        logging.basicConfig(filename='extractor_executions.log', 
                            encoding='utf-8',
                            level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', 
                            datefmt='%m/%d/%Y %I:%M:%S %p'
                            )
        logging.info('Initializing the Logger')

        # Load the cfg file
        cfg = load_config(args, path_to_config)
        
        # Select GPU
        torch.cuda.set_device(1)
        print('Device GPU: ', torch.cuda.current_device())
        
        # Check if the cfg file is for test
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)
        else:
            raise Exception("This function can only get features, classification \
            is not implemented. Please change TEST.ENABLE to True.")

if __name__ == "__main__":
    main()