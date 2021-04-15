from functools import wraps
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

EPOCH_DEPRECATION_ERROR = (
    "If the cause of this exception is the deprecation of `epoch` argument, "
    "set argument `deprecate_epoch=True` in `scheduler.__init__()` "
    "in which batch-level adjustment is sustained for the warmup scheduler "
    "while epoch-level adjustment is used for the main_scheduler abiding by "
    "the deprecation."
)


def suppress_warning(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning) 
            # safely ignore deprecation warning
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                raise Exception(EPOCH_DEPRECATION_ERROR) from ex
    return decorated


class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class LinearWarmupScheduler(object):
    """Modification of https://github.com/ildoonet/pytorch-gradual-warmup-lr.
    This scheduler resolves user warning issues in the lastest Pytorch version
    and includes modified methods for saving & loading state_dict.
    """
    def __init__(self, optimizer, main_scheduler, max_steps, max_epochs,
                 deprecate_epoch=False):
        if type(main_scheduler) == ReduceLROnPlateau:
            raise NotImplementedError('Not supported yet!')
        self.max_epochs = max_epochs
        self.last_epoch = 0
        self.epoch_per_step = 1 / max_steps
        self.eps = self.epoch_per_step * 0.1  # arbitrary smaller value
        self.main_scheduler = main_scheduler
        self.deprecate_epoch = deprecate_epoch
        
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        
        # Initialize epoch and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], 
                                 optimizer.param_groups))
        
    @property
    def _is_epoch_level_update(self):
        return (self.last_epoch > self.max_epochs and 
                self.last_epoch % 1.0 < self.eps)
        
    def get_last_lr(self):
        """Return last computed learning rate by current scheduler.
        """
        if not hasattr(self, '_last_lr'):
            raise RuntimeError("Run scheduler.step() first!")
        return self._last_lr
        
    def get_lr(self):
        ratio = min(self.last_epoch / self.max_epochs, 1.0)
        return [base_lr * ratio for base_lr in self.base_lrs]
    
    def _step(self, epoch=None):

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]                
            
    @suppress_warning
    def step(self):
        """step method for batch-level adjustment."""
        epoch = self.last_epoch + self.epoch_per_step
        if epoch > self.max_epochs:
            if self.deprecate_epoch:
                if self._is_epoch_level_update:
                    self.main_scheduler.step()
            else:
                self.main_scheduler.step(epoch - self.max_epochs)
            self.last_epoch = epoch
            self._last_lr = self.main_scheduler.get_last_lr()
        else:
            return self._step(epoch)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() 
                      if key not in ['optimizer', 'main_scheduler']}
        state_dict.update({"main_scheduler": self.main_scheduler.state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        self.main_scheduler.load_state_dict(state_dict.pop("main_scheduler"))
        self.__dict__.update(state_dict)
