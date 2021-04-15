from collections import defaultdict
from copy import deepcopy
import glob
import logging
import os
from shutil import copyfile
import subprocess

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate

from augment.transforms import ImageWithTransInfo
from distributed import comm

log = logging.getLogger('main')

OUT_DIR = 'runs'


def turn_into_debug_config(cfg):
    cfg.train.max_epochs = 2
    cfg.train.warmup_epochs = 1
    cfg.train.valid_interval = 1
    cfg.train.snapshot_interval = 1
    cfg.train.valid_online = True
    cfg.train.num_workers = 4
    cfg.eval.max_epochs = 2
    cfg.eval.warmup_epochs = 1
    cfg.eval.valid_interval = 1
    cfg.eval.num_workers = 4
    return cfg


def unwrap_if_distributed(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module


def wrap_if_distributed(module, device):
    if comm.get_world_size() > 1 and len(list(module.parameters())) > 0:
        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        module = DistributedDataParallel(module=module, 
                                         device_ids=[device], 
                                         broadcast_buffers=False, 
                                         find_unused_parameters=True)
    return module


def get_auto_save_dir(config_name, top_dir='', out_dir=OUT_DIR, 
                      n_zfill=3, overwrite=False):
    suffix_num = 0
    dir_ = os.path.join(top_dir, out_dir, config_name)
    _get_suffixed_dir = lambda n: f"{dir_}_" + str(suffix_num).zfill(n_zfill)
    while os.path.exists(_get_suffixed_dir(suffix_num)):
        suffix_num += 1
    if overwrite:
        suffix_num -= 1
    return _get_suffixed_dir(suffix_num)


def create_save_dir(save_dir, backup_files=(), save_git_head_hash=True):
    try:
        os.makedirs(save_dir)
        log.info(f"[Save] create save_dir: {save_dir}")
    except FileExistsError:
        log.warning(f"[Save] save_dir already exists: {save_dir}")
    for source in backup_files:
        target = os.path.join(save_dir, os.path.basename(source))
        copyfile(source, target)
        log.info(f"[Save] backup file: {target}")
    if save_git_head_hash:
        process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], 
                                   shell=False, stdout=subprocess.PIPE)
        git_head_hash = process.communicate()[0].strip().decode()
        hash_file = os.path.join(save_dir, "git_head_hash")
        with open(hash_file, 'a+') as f:
            f.write(git_head_hash + '\n')
        log.info(f"[Save] save git hash [{git_head_hash[:7]}] to {hash_file}")


def sync_weighted_mean(value, n_samples):
    comm.synchronize()
    value = comm.gather(value, dst=0)
    n_samples = comm.gather(n_samples, dst=0)
    
    if comm.get_rank() == 0:
        total_n = 0
        weighted_a = 0
        for a, n in zip(value, n_samples):
            total_n += n
            weighted_a += a * n
        w_mean = weighted_a / total_n
    else:
        w_mean = 0.0  # to make empty output placeholder
        
    w_mean = comm.scatter(w_mean, src=0)
    return w_mean
    

def decompose_collated_batch(collated_batch):
    batch_views = []
    batch_transf = []
    batch_ratio = []
    batch_size = []
    if isinstance(collated_batch, ImageWithTransInfo):
        collated_batch = [collated_batch]
    for x in collated_batch:
        image, transf, ratio, size = x.image, x.transf, x.ratio, x.size
        batch_views.append(image)
        transf = torch.cat(transf).reshape(len(transf), image.size(0))
        transf = torch.transpose(transf, 1, 0)
        batch_transf.append(transf)
        batch_ratio.append(ratio)
        batch_size.append(size)
    return batch_views, batch_transf, batch_ratio, batch_size
