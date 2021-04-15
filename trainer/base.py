import copy
import errno
import os
import logging
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .helper import TensorBoardWriter
from .linear_eval import iter_eval_epoch, linear_eval_online, linear_eval_offline
from data import get_loaders_for_trainer
from models import Backbone
from models.heads import SingleLayerLinearHead, TwoLayerLinearHead
from optim import get_optimizer_and_scheduler
import utils


log = logging.getLogger('main')
C = utils.Colorer.instance()


def _unwrap(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module


def _regression_loss(x, y):
        # eps = 1e-6 if torch.is_autocast_enabled() else 1e-12
        x = F.normalize(x, p=2, dim=1) #, eps=eps)
        y = F.normalize(y, p=2, dim=1) #, eps=eps)
        return (2 - 2 * (x * y).sum(dim=1)).view(-1)


class BYOLBasedTrainer:
    """This trainer supports BYOL-like training framework that can be subclassed 
    by other task-specific trainer classes. To specify a detailed algorithm, 
    the user should implement Traniner.run().
    """
    def __init__(self, cfg, online_network, target_network, 
                 predictor=None, evaluator=None,
                 train_loader=None, eval_loader=None):
        if cfg.train.enabled:
            assert train_loader is not None
            assert predictor is not None
        if cfg.train.enabled and cfg.train.online_eval:
            assert eval_loader is not None
            assert evaluator is not None
            
        self._modules = {}
        self._saving_targets = {}
        self.cfg = cfg
        self.device = cfg.device
        
        self.online_network = online_network
        self.target_network = target_network
        self.predictor = predictor
        self.evaluator = evaluator
        self.xent_loss = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self._setup_device_and_distributed_parallel(cfg.device)

        self.cur_epoch = 0
        self.max_epochs = 0
        self.max_eval_score = 0.
        self.max_eval_epoch = 0
        
        if self.cfg.train.enabled:
            self.m_base = self.m = cfg.train.m
            self.max_epochs = cfg.train.max_epochs
            self.total_global_step = len(train_loader) * cfg.train.max_epochs
            self.optimizer, self.scheduler = get_optimizer_and_scheduler(
                cfg=self.cfg, mode='train', modules=self._modules, loader=train_loader,
                exclude_from_lars=True, module_black_list=['target_network'])
            self.scaler = torch.cuda.amp.GradScaler() #init_scale=2**14)
            # default init_scale 2**16 will yield invalid gradient in the first interation 
            self.tb_writer = TensorBoardWriter.init_for_train_from_config(cfg)
        else:
            self.optimizer, self.scheduler, self.scaler = None, None, None

    def __setattr__(self, name, value):
        if hasattr(value, 'state_dict') and callable(value.state_dict):
            self._saving_targets[name] = value  # including optimzers & schedulers
        if isinstance(value, nn.Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)
        
    def run(self):
        """Main training algorithm should be implemented in this method."""
        raise NotImplementedError()
    
    @classmethod
    def init_from_config(cls, cfg):
        train_loader, eval_loader, num_classes = get_loaders_for_trainer(cfg)
        online_network = Backbone.init_from_config(cfg)
        target_network, predictor, evaluator = None, None, None
        if cfg.train.enabled:
            target_network = Backbone.init_from_config(cfg)
            predictor = TwoLayerLinearHead.init_predictor_from_config(cfg)
            evaluator = SingleLayerLinearHead.init_evaluator_from_config(
                cfg, num_classes)
        return cls(
            cfg=cfg,
            train_loader=train_loader,
            eval_loader=eval_loader,
            online_network=online_network,
            target_network=target_network,
            predictor=predictor,
            evaluator=evaluator,
        )

    def _setup_device_and_distributed_parallel(self, device):
        for name, module in self._modules.items():
            module = module.to(device)
            module = utils.wrap_if_distributed(module, device)
            self._modules[name] = module
            object.__setattr__(self, name, module)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), 
                                    self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _decay_ema_momentum(self, step):
        self.m = (1 - (1 - self.m_base) * 
                  (math.cos(math.pi * step / self.total_global_step) + 1) / 2)

    @staticmethod
    def _criterion(p_online, p_target):
        """Regression loss used in BYOL."""
        p_online_v1, p_online_v2 = p_online.chunk(2)
        p_target_v1, p_target_v2 = p_target.chunk(2)
        assert p_online_v1.size(0) == p_online_v2.size(0)
        assert p_target_v1.size(0) == p_target_v2.size(0)
        assert p_online_v1.size(0) == p_target_v1.size(0)
        # symmetric loss
        loss = _regression_loss(p_online_v1, p_target_v2)
        loss += _regression_loss(p_online_v2, p_target_v1)
        return loss.mean()

    def _initialize_target_network(self, from_online):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), 
                                    self.target_network.parameters()):
            if from_online:
                param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def _save_checkpoint(self, tag):
        save_path = f"{self.cfg.save_dir}/checkpoint_" + str(tag) + ".pth"
        state_dict = {
            'tag': str(tag), 
            'epoch': self.cur_epoch,
            'max_eval_score': self.max_eval_score,
            'max_eval_epoch': self.max_eval_epoch,
            }
        for key, target in self._saving_targets.items():
            if self.cfg.fake_checkpoint:
                target = "fake_state_dict"
            else:
                target = utils.unwrap_if_distributed(target)
                target = target.state_dict()
            state_dict[f"{key}_state_dict"] = target

        torch.save(state_dict, save_path)
        suffix = (C.debug(" (fake_checkpoint)") 
                  if self.cfg.fake_checkpoint else "")
        return save_path + suffix

    def save_checkpoint(self, epoch):
        save_path = self._save_checkpoint(str(epoch))
        log.info(f"[Save] restore the model's checkpoint: {save_path}")
        return save_path
    
    def save_best_checkpoint(self):
        save_path = self._save_checkpoint('best')
        log.info(f"[Save] restore the best model's checkpoint: {save_path}")
        return save_path

    def symlink_checkpoint_with_tag(self, epoch, tag):
        save_path = f"{self.cfg.save_dir}/checkpoint_{epoch}.pth"
        symlink_path = f"{self.cfg.save_dir}/checkpoint_{tag}.pth"
        if not os.path.exists(save_path):
            self._save_checkpoint(epoch)
        try:
            os.symlink(os.path.abspath(save_path), symlink_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(symlink_path)
                os.symlink(os.path.abspath(save_path), symlink_path)
            else:
                raise e
        finally:
            log.info(f"[Save] make a symlink of the current model: "
                     f"{symlink_path}")
        return symlink_path

    def load_checkpoint_if_available(self, tag='last'):
        if self.cfg.overwrite:
            assert not self.cfg.load_dir, \
                "Mutually exclusive aruguements: overwrite, load_dir."
            log.warning("Overwrite checkpoints in save_dir.")
            return False
        try:
            load_dir = self.cfg.load_dir or self.cfg.save_dir
            load_path = f"{load_dir}/checkpoint_{tag}.pth"
            state_dict = torch.load(load_path)
        except FileNotFoundError:
            if self.cfg.load_dir:
                raise FileNotFoundError(f"Can't find checkpoint at {load_dir}")
            else:
                log.warning(f'No checkpoint to resume from {load_dir}.')
            return False

        self.cur_epoch = state_dict['epoch']
        self.max_eval_score = state_dict['max_eval_score']
        self.max_eval_epoch = state_dict['max_eval_epoch']
        state_dict = {k[:-len('_state_dict')]: v for k, v in state_dict.items() 
                      if k.endswith('_state_dict')}
        log.info(f"[Resume] Loaded chekpoint (epoch: {self.cur_epoch}) "
                 f"from: {load_path}")

        missing_keys = set(self._saving_targets.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(self._saving_targets.keys())
        assert len(missing_keys) == 0, "Missing keys!"
        log.info("[Resume] Redundant keys: "
                 f"{list(unexpected_keys) if unexpected_keys else 'None'}")

        for key, target in self._saving_targets.items():
            if state_dict[key] == 'fake_state_dict':
                log.info(f"[Resume] Loaded {key}: {C.debug('(fake_chekpoint)')}")
            else:
                kwargs = {'strict': False} if isinstance(target, nn.Module) else {}
                loaded = _unwrap(target).load_state_dict(state_dict[key], **kwargs)
                if isinstance(target, nn.Module):
                    assert len(loaded.missing_keys) == 0
                if isinstance(target, Backbone):
                    # the projector is be ignored in evaluation-only cases
                    assert all([key.startswith('projector.') 
                                for key in loaded.unexpected_keys])
                log.info(f"[Resume] Loaded {key}")
        return True
