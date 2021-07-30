# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from collections import abc, OrderedDict
import logging
import os
import yaml

from .swin_xfmr import SwinTransformer
from distributed import comm
from utils.config import Config, _update_dict

import torch

log = logging.getLogger('main')


CONFIG_MAP = {
    "swin-t": "models/swin_xfmr/swin_config/swin_tiny_patch4_window7_224.yaml",
    "swin-s": "models/swin_xfmr/swin_config/swin_small_patch4_window7_224.yaml",
    "swin-b": "models/swin_xfmr/swin_config/swin_base_patch4_window7_224.yaml",
    "swin-l": "models/swin_xfmr/swin_config/swin_large_patch4_window7_224.yaml",
}


def build_swin_xformer(name):
    config_file = CONFIG_MAP[name]
    config = load_config_yaml(config_file)
    config = Config(config)
    config.freeze()
    return SwinTransformer(
        pretrain_img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )


def load_config_yaml(cfg_file, config=None):
    if config is None:
        config = OrderedDict()
    
    with open(cfg_file, 'r') as f:
        config_src = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in config_src.setdefault('BASE', ['']):
        if cfg:
            load_config_yaml(
                os.path.join(os.path.dirname(cfg_file), cfg), config
            )
    log.info(f'[SwinXFMR] merge config from {cfg_file}')
    _update_dict(config, config_src)
    return config
