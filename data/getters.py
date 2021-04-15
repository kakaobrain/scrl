import numpy as np

from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from .datasets import get_dataset
from .dataloader import FastDataloader
from augment import (get_transforms, get_center_crop_transforms, 
                     get_simple_transforms)
from distributed import comm


DEBUG_NUM_BATCH = 2


def _collate_fn(batch):
    imgs, targets = default_collate(batch)
    if len(imgs) == 1:
        # squeeze single view dim
        imgs = imgs[0]
    return imgs, targets


def get_loaders_for_trainer(cfg):
    train_loader, eval_loader, num_classes = (None,) * 3
    
    # train dataset
    if cfg.train.enabled:
        if cfg.debug:
            n_samples = comm.get_world_size() * DEBUG_NUM_BATCH
            n_samples = n_samples * cfg.train.batch_size_train
        else:
            n_samples = -1
            
        train_dataset, num_classes = get_dataset(
            data_name=cfg.dataset.name,
            data_root=cfg.dataset.root,
            train=True,
            transform=get_transforms(cfg, train=True),
            num_subsample=int(n_samples),
        )
        train_sampler = DistributedSampler(
            dataset=train_dataset, rank=comm.get_rank(),
            num_replicas=comm.get_world_size(), shuffle=True
        )
        train_loader = FastDataloader(
            dataset=train_dataset, batch_size=cfg.train.batch_size_train,
            num_workers=cfg.train.num_workers, sampler=train_sampler,
            drop_last=False, collate_fn=_collate_fn
        )

    # test dataset (for online evaluation)
    if cfg.train.enabled and cfg.train.online_eval:
        if cfg.debug:
            n_samples = comm.get_world_size() * DEBUG_NUM_BATCH
            n_samples = n_samples * cfg.train.batch_size_eval
            
        eval_dataset, num_classes = get_dataset(
            data_name=cfg.dataset.name,
            data_root=cfg.dataset.root,
            train=False,
            transform=get_transforms(cfg, train=False),
            num_subsample=int(n_samples),
        )
        eval_sampler = DistributedSampler(
            dataset=eval_dataset, rank=comm.get_rank(),
            num_replicas=comm.get_world_size(), shuffle=True
        )
        eval_loader = FastDataloader(
            dataset=eval_dataset, batch_size=cfg.train.batch_size_eval,
            num_workers=cfg.train.num_workers, sampler=eval_sampler,
            drop_last=False, collate_fn=_collate_fn
        )
        
    return train_loader, eval_loader, num_classes


def get_loaders_for_linear_eval(cfg):
    if cfg.debug:
        train_n_samples = comm.get_world_size() * DEBUG_NUM_BATCH
        train_n_samples = train_n_samples * cfg.eval.batch_size_train
        eval_n_samples = comm.get_world_size() * DEBUG_NUM_BATCH
        eval_n_samples = eval_n_samples * cfg.eval.batch_size_eval
    else:
        train_n_samples = eval_n_samples = -1
    
    # augmentation
    train_transforms = get_simple_transforms(
        input_size=cfg.augment.input_size
    )
    eval_transforms = get_center_crop_transforms(
        input_size=cfg.augment.input_size
    )
    
    # dataset
    train_dataset, num_classes = get_dataset(
        data_name=cfg.dataset.name, 
        data_root=cfg.dataset.root,
        train=True, 
        transform=train_transforms,
        num_subsample=int(train_n_samples),
    )
    eval_dataset, _ = get_dataset(
        data_name=cfg.dataset.name, 
        data_root=cfg.dataset.root,
        train=False, 
        transform=eval_transforms,
        num_subsample=int(eval_n_samples),
    )

    # sampler
    train_sampler = DistributedSampler(
        dataset=train_dataset, rank=comm.get_rank(),
        num_replicas=comm.get_world_size(), shuffle=True
    )
    eval_sampler = DistributedSampler(
        dataset=eval_dataset, rank=comm.get_rank(),
        num_replicas=comm.get_world_size(), shuffle=True
    )

    # dataloader
    num_workers = cfg.eval.num_workers if not cfg.debug else 4
    train_loader = FastDataloader(
        dataset=train_dataset, batch_size=cfg.eval.batch_size_train,
        num_workers=num_workers, drop_last=False, 
        sampler=train_sampler, collate_fn=_collate_fn
    )
    eval_loader = FastDataloader(
        dataset=eval_dataset, batch_size=cfg.eval.batch_size_eval,
        num_workers=num_workers, drop_last=False, 
        sampler=eval_sampler, collate_fn=_collate_fn
    )
    return train_loader, eval_loader, num_classes
