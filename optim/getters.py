from copy import deepcopy
from itertools import chain
import logging

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from .scheduler import LinearWarmupScheduler
from .optim import CustomLARS
from distributed import comm

log = logging.getLogger('main')


def bisect_params_with_names(named_params, criterion):
    """Bisect name_parameters according to whether their names include any of 
    criterion keywords. This can be used to separate bias and batch norm 
    parameters in order to treat them differently during the internal process of 
    LARS optimizer.
    """
    out_params = []
    in_params = []
    for name, param in named_params:
        if any([c in name for c in criterion]):
            in_params.append(param)
        else:
            out_params.append(param)
    return out_params, in_params


def scale_learning_rate(cfg, mode):
    assert mode in ['train', 'eval']
    world_size = comm.get_world_size()
    lr_origin = cfg[mode].optim.lr
    local_batch = cfg[mode].batch_size_train
    global_batch = int(local_batch * world_size)
    ratio = global_batch / 256.
    lr = lr_origin * ratio
    log.info(f'[LR({mode})] local_batch ({local_batch}) x '
             f'world_size ({world_size}) = global_batch ({global_batch})')
    log.info(f'[LR({mode})] scale LR from {lr_origin} '
             f'to {lr} (x{ratio:3.2f}) by linear scaling rule.')
    optim = deepcopy(cfg[mode].optim)
    optim.lr = lr
    return optim


def get_optimizer_and_scheduler(cfg, mode, modules, loader, 
                                exclude_from_lars=False, module_black_list=(),):
    assert mode in ['train', 'eval']
    modules = [mod for name, mod in modules.items() 
               if name not in module_black_list]

    if exclude_from_lars:
        # separate batch_norm & bias params to exclude them 
        # from lars adaption & weight decaying (as in official code)
        params_default, params_bn_bias = bisect_params_with_names(
            named_params=chain(*[m.named_parameters() for m in modules]),
            criterion=['bn', 'bias'],
        )
    else:
        params_default = chain(*[m.parameters() for m in modules])
        params_bn_bias = None
        
    param_group = [{'params': params_default, 
                    'lars_adaptation': True}]
    if params_bn_bias:
        param_group.append({'params': params_bn_bias, 
                            'lars_adaptation': False, 
                            'weight_decay': 0.})
        
    # optimizer
    optimizer = CustomLARS(torch.optim.SGD(
        param_group, **scale_learning_rate(cfg, mode)))
    
    # scheduler
    t_max = cfg[mode].max_epochs - cfg[mode].warmup_epochs
    scheduler = CosineAnnealingLR(optimizer=optimizer, 
                                  T_max=t_max, 
                                  eta_min=0.)
    if cfg[mode].warmup_epochs:
        scheduler = LinearWarmupScheduler(
            optimizer=optimizer, 
            max_steps=len(loader),
            max_epochs=cfg[mode].warmup_epochs, 
            main_scheduler=scheduler,
            deprecate_epoch=False)
    
    return optimizer, scheduler
