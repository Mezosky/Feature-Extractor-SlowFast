import math
import torch
from fvcore.common.registry import Registry
from torch.distributed.algorithms.ddp_comm_hooks import (
    default as comm_hooks_default,)

import slowfast.utils.logging as logging
from slowfast.models.video_model_builder import _POOL1, ResNet

from .video_model_builder import MvitFeat, ResnetFeat, SlowFastFeat, X3DFeat
from .head_helper import ResNetBasicHead, TransformerBasicHead, X3DHead

# logger = logging.get_logger(__name__)

# MODEL_REGISTRY = Registry("MODEL")
# MODEL_REGISTRY.__doc__ = """
# Registry for video model.
# The registered object will be called with `obj(cfg)`.
# The call should return a `torch.nn.Module` object.
# """


def build_model(cfg, gpu_id=None):
    """
    Builds the models to get features from videos.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone.
        gpu_id (Optional[int]): specify the gpu index to build model.
    Return:
        model: model loaded and configurated for the feature extraction.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    name = cfg.MODEL.MODEL_NAME
    print(f"Loading model... {name}")
    
    if name == "ResNet":
        pool_size = _POOL1[cfg.MODEL.ARCH]
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        model = ResnetFeat(cfg)
        model.head = ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
                )

    elif name == "SlowFast":
        pool_size = _POOL1[cfg.MODEL.ARCH]
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        model = SlowFastFeat(cfg)
        model.head = ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
                )
    
    elif name == "X3D":
        model = X3DFeat(cfg)
        dim_out = model.head.conv_5.in_channels
        dim_inner = model.head.conv_5.out_channels
        spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
        model.head = X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
                )
    
    elif name == "MViT":
        model = MvitFeat(cfg)

    else:
        raise Exception("You have not specified a video model.")

    print("comprobate BN.NORM_TYPE",cfg.BN.NORM_TYPE)
    if cfg.BN.NORM_TYPE == "sync_batchnorm_apex":
        try:
            import apex
        except ImportError:
            raise ImportError("APEX is required for this model, please install")

        logger.info("Converting BN layers to Apex SyncBN")
        process_group = apex.parallel.create_syncbn_process_group(
            group_size=cfg.BN.NUM_SYNC_DEVICES
        )
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group
        )

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True
            if cfg.MODEL.DETACH_FINAL_FC
            or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            else False,
        )
        if cfg.MODEL.FP16_ALLREDUCE:
            model.register_comm_hook(
                state=None, hook=comm_hooks_default.fp16_compress_hook
            )

    return model