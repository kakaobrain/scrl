from contextlib import ExitStack
import logging
import os

import torch
from torch import nn
from torch.autograd import enable_grad
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from data import get_loaders_for_linear_eval
from distributed import comm
from models import Backbone, SingleLayerLinearHead
from optim import get_optimizer_and_scheduler
from utils import sync_weighted_mean, unwrap_if_distributed, Config, Colorer
from utils import TimeOfArrivalEstimator, DistributedProgressDisplayer
from utils import ExceptionLogger

log = logging.getLogger('main')
C = Colorer.instance()


def _get_desc(mode, save_dir):
    head = C.selected(f'[{mode}]')
    save_dir = C.underline(f'[{save_dir}]')
    return f"{head} {save_dir}"


@ExceptionLogger("error")
@torch.no_grad()  # the order \btw decorators matters
def iter_eval_epoch(
    cfg: Config, backbone: Backbone, head: SingleLayerLinearHead, 
    loader: DataLoader, criterion=None, finetune=False):
    """Work on evaluation mode when criterion=None."""
    
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        labels = y.to(cfg.device, non_blocking=True)
        
        # feedforward
        with autocast(not cfg.disable_autocast):
            
            with enable_grad() if criterion and finetune else ExitStack():
                h = backbone(x, boxes=None, no_projection=True)
                
            with enable_grad() if criterion else ExitStack():
                logits = head(h if finetune else h.detach())
                loss = criterion(logits, labels) if criterion else 0.
                
        yield logits, labels, loss
        

@ExceptionLogger("error")
def linear_eval_online(cfg: Config, epoch: int, eval_loader: DataLoader, 
                       backbone: Backbone, head: SingleLayerLinearHead):
    assert eval_loader is not None
    assert head is not None
    backbone.eval()
    n_correct = 0
    n_samples = 0
    
    with DistributedProgressDisplayer(
            max_steps=len(eval_loader), 
            no_pbar=cfg.no_pbar,
            desc=_get_desc('Upstream: Eval', cfg.save_dir)
        ) as disp:
    
        for logits, labels, _ in iter_eval_epoch(
                cfg=cfg,backbone=backbone, head=head, 
                loader=eval_loader, criterion=None
            ):
            preds = torch.argmax(logits, dim=1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
            acc = (preds == labels).float().mean().item() * 100
            
            disp.update_with_postfix(
                f"ep:{epoch}/{cfg.train.max_epochs}, "
                f"#samples:{n_samples:5d}, "
                f"Acc: {acc:5.4f}"
            )

    backbone.train()
    acc = n_correct / n_samples * 100
    acc = sync_weighted_mean(acc, n_samples)

    return acc
        

@ExceptionLogger("error")
def linear_eval_offline(cfg: Config, backbone: nn.Module, finetune=False):
    train_loader, eval_loader, num_classes = get_loaders_for_linear_eval(cfg)
    head = SingleLayerLinearHead.init_evaluator_from_config(cfg, num_classes)
    head = head.to(unwrap_if_distributed(backbone).device)
    if comm.get_local_size() > 1:
        head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
        head = DistributedDataParallel(module=head, 
                                       device_ids=[cfg.device], 
                                       broadcast_buffers=False, 
                                       find_unused_parameters=True)
    modules = {'head': head}
    if finetune:
        log.info("Fine-tuning mode.")
        backbone.train()
        modules.update({'backbone': backbone})
    else:
        # to use running statistics for the frozen backbone
        backbone.eval()
        
    optimizer, scheduler = get_optimizer_and_scheduler(
        cfg=cfg, mode='eval', modules=modules, loader=train_loader)
    scaler = torch.cuda.amp.GradScaler()

    max_eval_acc = 0.
    max_eval_epoch = 0
    max_epochs = cfg.eval.max_epochs
    criterion = torch.nn.CrossEntropyLoss()
    eta = TimeOfArrivalEstimator.init_from_epoch_steps(
        epochs=cfg.eval.max_epochs,
        epoch_steps=len(train_loader),
    )
    
    # training & validation loop
    for epoch in range(1, max_epochs + 1):
        train_loader.sampler_origin.set_epoch(epoch)
            
        with DistributedProgressDisplayer(
                max_steps=len(train_loader), 
                no_pbar=cfg.no_pbar,
                desc=_get_desc('Downstream:Train', cfg.save_dir)
            ) as disp:
            
            for logits, labels, loss in iter_eval_epoch(
                    cfg=cfg, backbone=backbone, head=head, loader=train_loader, 
                    criterion=criterion, finetune=finetune
                ):
                scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item() * 100
                comm.synchronize()

                disp.update_with_postfix(
                    f"ep:{epoch}/{max_epochs}, "
                    f"lr:{scheduler.get_last_lr()[0]:5.4f}, "
                    f"Loss:{loss:5.4f}, "
                    f"Acc:{acc:5.2f}%, "
                    f"EvalMax:{max_eval_acc:5.2f}%, "
                    f"eta:{eta.estimate_step_str()}"
                )

        # end of each epoch
        del logits, labels, loss

        is_valid_step = (cfg.eval.valid_interval 
                         and epoch % cfg.eval.valid_interval == 0)
        if not (is_valid_step or epoch == max_epochs) or epoch < 1:
            continue

        # validation
        n_correct = 0
        n_samples = 0
        backbone.eval()
        
        with DistributedProgressDisplayer(
                max_steps=len(eval_loader), 
                no_pbar=cfg.no_pbar,
                desc=_get_desc('Downstream: Eval', cfg.save_dir),
            ) as disp:
        
            for logits, labels, _ in iter_eval_epoch(
                    cfg=cfg, backbone=backbone, head=head, 
                    loader=eval_loader, criterion=None
                ):
                
                preds = torch.argmax(logits, dim=1)
                n_samples += labels.size(0)
                n_correct += (preds == labels).sum().item()
                acc = (preds == labels).float().mean().item() * 100
                
                disp.update_with_postfix(
                    f"ep:{epoch}/{max_epochs}, "
                    f"#samples:{n_samples:5d}, "
                    f"Acc:{acc:5.2f}%, "
                    f"EvalMax:{max_eval_acc:5.2f}%"
                )
                    
        last_eval_acc = n_correct / n_samples * 100
        last_eval_acc = sync_weighted_mean(last_eval_acc, n_samples)

        is_best = ""
        if last_eval_acc > max_eval_acc:
            max_eval_acc = last_eval_acc
            max_eval_epoch = epoch
            is_best = C.red("[<- Best Acc.]")

        if comm.synchronize() and comm.is_local_main_process():
            log.info(
                f"{C.red('[Eval result]')} "
                f"ep:{epoch}/{max_epochs}, "
                f"EvalAcc:{last_eval_acc:5.2f}%, "
                f"EvalMax:{max_eval_acc:5.2f}% {is_best}"
            )
        
        if finetune:
            backbone.train()

    return last_eval_acc, max_eval_acc, max_eval_epoch
