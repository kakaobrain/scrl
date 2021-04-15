import gc
import logging

import torch
import torch.nn.functional as F

from .base import BYOLBasedTrainer
from .helper import TrainerOutputs, Loss, Metric, TaskState, TaskReturns
from .linear_eval import linear_eval_online, linear_eval_offline
from .result_pack import ResultPack
from distributed import comm
from models import SCRLBoxGenerator
from models.heads import SingleLayerLinearHead, TwoLayerLinearHead
from utils import detect_anomaly_with_redirected_cpp_error
from utils import DistributedProgressDisplayer, decompose_collated_batch
from utils import Colorer, ColorerContext, TimeOfArrivalEstimator

torch.backends.cudnn.benchmark = True
log = logging.getLogger('main')
log_result = logging.getLogger('result')
C = Colorer.instance()


def _release_memory(*objects):
    del objects
    gc.collect()
    torch.cuda.empty_cache()


class SCRLTrainer(BYOLBasedTrainer):
    """SCRL trainer that subclasses BYOLBasedTrainer. 
    Refer to base.BYOLBasedTrainer for the inherited attributes.
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if self.cfg.network.scrl.enabled:
            self.box_generator = SCRLBoxGenerator.init_from_config(cfg)
        
    def run(self):
        log.info(C.green("[!] Start the Trainer."))
        result_pack = ResultPack(exp_name=self.cfg.config_name)
        
        if self.target_network is not None:
            self._initialize_target_network(from_online=False)
            
        # resume from the chekcpoint if possible
        self.load_checkpoint_if_available()
        if self.cfg.train.enabled and self.cur_epoch == self.max_epochs:
            self.cfg.defrost()
            self.cfg.train.enabled = False
            self.cfg.freeze()

        if not self.cfg.train.enabled:
            self.max_epochs = 0
            log.info(C.green('[!] Load Pre-trained Parameters.'))
        else:
            method = 'SCRL' if self.cfg.network.scrl.enabled else 'BYOL'
            log.info(C.green(f"[!] Upstream: {method} Pre-training."))
            eta = TimeOfArrivalEstimator.init_from_epoch_steps(
                epochs=self.max_epochs - self.cur_epoch,
                epoch_steps=len(self.train_loader),
            )
                
        comm.synchronize()
        # Upstream: BYOL or SCRL
        for epoch in range(1 + self.cur_epoch, self.max_epochs + 1):
            self.cur_epoch = epoch
            self.train_loader.sampler_origin.set_epoch(epoch)
            disp = DistributedProgressDisplayer(
                max_steps=len(self.train_loader), 
                no_pbar=self.cfg.no_pbar,
                desc=(f"{C.selected('[Upstream:Train]')} "
                      f"{C.underline(f'[{self.cfg.save_dir}]')}")
            )
                
            for step, (views, labels) in enumerate(self.train_loader, start=1):
                
                # gear up inputs (and spatially consistent boxes if needed)
                if self.cfg.network.scrl.enabled:
                    views, transf, _, _ = decompose_collated_batch(views)
                    boxes = self.box_generator.generate(transf)
                else:
                    boxes =  None
                    
                # model forward and loss computation
                with detect_anomaly_with_redirected_cpp_error(self.cfg.detect_anomaly):
                    with torch.cuda.amp.autocast(not self.cfg.disable_autocast):
                        outs = self._forward(views, labels, boxes)
                        loss_total = outs.by_cls_name('Loss').weighted_sum_scalars()
                        
                # optimization
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.scaler.scale(loss_total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # log
                disp.update_with_postfix(
                    f"ep:{epoch}/{self.max_epochs}, "
                    f"lr:{self.scheduler.get_last_lr()[0]:5.4f}, "
                    f"m:{self.m:5.4f}, "
                    f"{str(outs.by_cls_name('Loss'))}, "
                    f"{str(outs.by_cls_name('Metric'))}, "
                    f"eEvaMax:{self.max_eval_score:5.2f}%, "
                    f"eta:{eta.estimate_step_str()}"
                )
                global_step = len(self.train_loader) * (epoch - 1) + step - 1
                self.tb_writer.add_outputs(outs, global_step)
                
                # EMA update
                comm.synchronize()
                if self.target_network is not None:
                    self._decay_ema_momentum(global_step)
                    self._update_target_network_parameters()

            # end of each epoch
            disp.close()
                
            # save model on a regular basis
            if (comm.is_main_process() and 
                    (epoch % self.cfg.train.snapshot_interval == 0 or 
                     epoch == self.max_epochs)):
                self.save_checkpoint(epoch)
                self.symlink_checkpoint_with_tag(epoch, 'last')

            # online evaluation
            if self.cfg.train.online_eval:
                last_eval_on = linear_eval_online(
                    cfg=self.cfg,
                    epoch=self.cur_epoch,
                    eval_loader=self.eval_loader,
                    backbone=self.online_network,
                    head=self.evaluator)
                
                is_best = ""
                if (last_eval_on and last_eval_on > self.max_eval_score):
                    self.max_eval_score = last_eval_on
                    self.max_eval_epoch = epoch 
                    self.save_best_checkpoint()
                    # self.symlink_best_checkpoint(epoch)
                    is_best = C.red("[<- Best Acc.]")
            
                log.info(
                    f"{C.red('[Eval result]')} "
                    f"ep:{epoch}/{self.max_epochs}, "
                    f"eEvalAcc:{last_eval_on:5.2f}%, "
                    f"eEvaMax:{self.max_eval_score:5.2f}% {is_best}"
                )

            if comm.synchronize() and comm.is_local_main_process():
                result_dict = outs.scalar_only().to_dict()
                result_dict.update({'eEvaMax': self.max_eval_score})
                result_pack.append(epoch, **result_dict)
                yield TaskReturns(state=TaskState.TRAIN, 
                                  value={'pack': result_pack})

        # end of upstream
        if self.cfg.train.enabled and comm.is_local_main_process():
            final_result = (
                f"[Final result (pre-training)] [{self.cfg.save_dir}] "
                f"{str(outs.by_cls_name('Loss'))}, "
                f"eEvalAcc:{last_eval_on:5.3f}%, "
                f"eEvaMax:{self.max_eval_score:5.3f}% @{self.max_eval_epoch}ep."
            )
            log.info(C.cyan(final_result))
            log_result.info(final_result)
            
        # Downstream: linear evaluation
        if self.cfg.eval.enabled:
            if comm.is_local_main_process():
                # yield the state here so as to update progress bar, etc., since
                # no returns there will be until evaluation is fully completed.
                # (Note that this comment is relvant to our internal API)
                yield TaskReturns(state=TaskState.EVAL)

            log.info(C.green(f"[!] Downstream: Linear Evaluation."))
            last_eval_off, max_eval_off, max_eval_epoch = linear_eval_offline(
                cfg=self.cfg,
                backbone=self.online_network,
                finetune=self.cfg.eval.finetune)
            
            if comm.synchronize() and comm.is_local_main_process():
                final_result = (
                    f"[Final result (linear eval.)] [{self.cfg.save_dir}] "
                    f"last:{last_eval_off:5.3f}%, "
                    f"max:{max_eval_off:5.3f}% @{max_eval_epoch}ep."
                )
                log.info(C.cyan(final_result))
                log_result.info(final_result)
            
            log.info("[Save] the final results are saved in: ./log_result.txt.")

        if comm.is_local_main_process():
            yield TaskReturns(state=TaskState.DONE) 

        if self.cfg.spawn_ctx:
            self.cfg.spawn_ctx.join()

        log.info(C.green("[!] End of the Trainer."))
        

    def _forward(self, views, labels, boxes=None):
        # compose mini-batch of views
        views = torch.cat(views, dim=0)
        views = views.to(self.device, non_blocking=True)

        # compose mini-batch of the matched box coords.
        if boxes is not None:
            boxes = torch.cat(boxes, dim=0)
            boxes = boxes.to(self.device, non_blocking=True)

        # online network 
        p_online, h_online = self.online_network(views, boxes)

        # return None by default
        byol_loss, scrl_loss, eval_loss, eval_acc = (None,) * 4
        
        # target network
        with torch.no_grad():
            p_target, _ = self.target_network(views, boxes)

        if self.cfg.network.scrl.enabled:
            # SCRL loss
            p_online = self.predictor(p_online)
            scrl_loss = self._criterion(p_online, p_target)
        else:
            # BYOL loss
            p_online = self.predictor(p_online)
            byol_loss = self._criterion(p_online, p_target)

        # online evaluator loss
        if self.cfg.train.online_eval:
            labels = torch.cat((labels,) * 2, dim=0)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.evaluator(h_online.detach())  # stop gradient
            eval_loss = self.xent_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)
            eval_acc = (preds == labels).float().mean().item()
            
        return TrainerOutputs(*[
            Loss(name='bLoss', value=byol_loss, fmt="4.3f"),
            Loss(name='sLoss', value=scrl_loss, fmt="4.3f"),
            Loss(name='eLoss', value=eval_loss, fmt="4.3f"),
            Metric(name='eAcc', value=eval_acc, fmt="5.2f", 
                   weight=100., suffix="%"),
        ])
