from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args #, load_config
from configs.custom_config import load_config
import torch

#from test_net import test
from get_features import test

def main():
    """
    Main function to process the videos.
    """
    args = parse_args()

    for path_to_config in args.cfg_files:

        cfg = load_config(args, path_to_config)

        torch.cuda.set_device(0)
        print('Device GPU: ', torch.cuda.current_device())
        
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()