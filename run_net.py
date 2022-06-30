from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import torch

from test_net import test

def main():
    """
    Main function to process the videos.
    """
    args = parse_args()

    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        print(args)
        print("#"*50)
        cfg = load_config(args, path_to_config)

        torch.cuda.set_device(1)
        print('Device GPU: ',torch.cuda.current_device())
        
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()