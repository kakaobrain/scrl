import torch
from torchlars import LARS
from torchlars._adaptive_lr import compute_adaptive_lr


class CustomLARS(LARS):
    """One can choose whether or not to use LARS adaptation w.r.t. different parameter groups
    by adding a key-value pair, i.e. {'lars_adaptation: boolean} to the param_group argument.
    """
    def __init__(self, *args, exclude_weight_decay=(), exclude_lars_adaptation=(), **kwargs):
        super(CustomLARS, self).__init__(*args, **kwargs)
        self._exclude_weigt_decay = exclude_weight_decay
        self._exclude_lars_adaptation = exclude_lars_adaptation

    def apply_adaptive_lrs(self, weight_decays):
        with torch.no_grad():
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    weight_decay = 0.0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if group.get('lars_adaptation', True):

                        param_norm = p.norm()
                        grad_norm = p.grad.norm()

                        # The optimizer class has no method to change `dtype` of
                        # its inner tensors (like `adaptive_lr`) and to select to
                        # use CPU or GPU with Tensor. LARS's interface follows the
                        # optimizer class's interface, so LARS cannot change
                        # `dtype` of inner tensors explicitly also. In that
                        # context, we have constructed LARS can modify its member
                        # variable's spec implicitly by comparing with given spec
                        # by the original optimizer's element.
                        param_norm_spec = (param_norm.is_cuda, param_norm.type())
                        adaptive_lr_spec = (self.adaptive_lr.is_cuda, self.adaptive_lr.type())

                        if param_norm_spec != adaptive_lr_spec:
                            self.adaptive_lr = torch.ones_like(param_norm)

                        # calculate adaptive lr & weight decay
                        adaptive_lr = compute_adaptive_lr(
                            param_norm,
                            grad_norm,
                            weight_decay,
                            self.eps,
                            self.trust_coef,
                            self.adaptive_lr)

                    else:
                        adaptive_lr = group['lr']

                    p.grad.add_(p.data, alpha=weight_decay)
                    p.grad.mul_(adaptive_lr)
