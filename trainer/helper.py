from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from distributed import comm


class TaskState(Enum):
    INIT = 1
    TRAIN = 2
    EVAL = 3
    DONE = 4
    
 
@dataclass   
class TaskReturns:
    state: TaskState
    value: Dict = None
        

class Output(object):
    def __init__(self, name, value=None, weight=1., fmt="5.4f", suffix=""):
        self._name = name
        self._value = value
        self._weight = weight
        self._fmt = fmt
        self._suffix = suffix
        try:
            self.__repr__()
        except ValueError:
            raise ValueError(f'Invalid format specifer: {fmt}')
        
    def __repr__(self):
        if self.value is None:
            return f"{self._name}:<empty value>"
        if (isinstance(self.value, torch.Tensor) and
                self.value.view(-1).size(0) > 1):
            return f"{self._name}:<none scalar value>"
        return f"{self._name}:{self.weighted_value:{self._fmt}}{self._suffix}"
    
    def show(self):
        return self.__repr__() + f" (x{self._weight})"
    
    def is_scalar(self):
        if (self.value is None or
                (isinstance(self.value, torch.Tensor) and 
                 self.value.view(-1).size(0) > 1)):
            return False
        return True
        
    @property
    def value(self):
        if self._value is None:
            return None
        return self._value
    
    @property
    def weighted_value(self):
        if self._value is None:
            return None
        return self._value * self._weight
    

class Loss(Output):
    def __init__(self, name, value=None, weight=1., fmt="5.4f", suffix=""):
        super(Loss, self).__init__(name, value, weight, fmt, suffix)
        
        
class Metric(Output):
    def __init__(self, name, value=None, weight=100., fmt="5.2f", suffix=""):
        super(Metric, self).__init__(name, value, weight, fmt, suffix)


class TrainerOutputs(object):
    """This class aids to extend trainer outputs without additional cost.
    """
    def __init__(self, *outputs):
        if not all([isinstance(output, Output) for output in outputs]):
            raise ValueError("positional arguments have to be subclasses of "
                             "trainer_output.Output.")
        if not len(outputs) == len(set([output._name for output in outputs])):
            raise ValueError("duplicated name of trainer_output.Output.")
        self._outputs = OrderedDict({output._name: output for output in outputs})
        
    def __repr__(self):
        return ", ".join([output.__repr__() for output in self._outputs.values()
                         if output.is_scalar()])
    
    def __getitem__(self, key):
        return self._outputs.__getitem__(key).weighted_value
    
    def __len__(self):
        return self._outputs.__len__()
    
    def __iter__(self):
        return self._outputs.__iter__()
    
    def __contains__(self, key):
        return key in self._outputs
    
    def __del__(self):
        del self._outputs
        
    def show(self):
        return ", ".join([output.show() for output in self._outputs.values()
                         if output.is_scalar()])
    
    def keys(self):
        return list(self._outputs.keys())
    
    def values(self):
        return [output.weighted_value for output in self._outputs.values()]
    
    def items(self):
        return list(zip(self.keys(), self.values()))
    
    def to_dict(self):
        return OrderedDict(**self)
    
    def by_cls_name(self, cls_name):
        outputs = [output for output in self._outputs.values() 
                   if output.__class__.__name__ == cls_name]
        if len(outputs) == 0:
            return None
        return TrainerOutputs(*outputs)
    
    def scalar_only(self):
        return TrainerOutputs(*[
            output for output in self._outputs.values() 
            if output.is_scalar()])

    def sum_scalars(self):
        return sum([output.value for output in self._outputs.values()
                    if output.is_scalar()])
    
    def weighted_sum_scalars(self):
        return sum([output.weighted_value for output in self._outputs.values()
                    if output.is_scalar()])


class TensorBoardWriter(object):
    def __init__(self, interval=1, save_dir=None, flush_secs=120):
        self.interval = interval
        if comm.is_main_process():
            self.writer = SummaryWriter(log_dir=f"tboard/{save_dir}/", 
                                        flush_secs=flush_secs)
        else:
            self.writer = None
                
    @classmethod
    def init_for_train_from_config(cls, cfg):
        return cls(
            interval=cfg.train.tb_interval,
            save_dir=cfg.save_dir,
            flush_secs=120 if not cfg.debug else 5,
        )
        
    def __del__(self):
        if self.writer is not None:
            self.writer.close()
        
    def add_outputs(self, outputs, global_step, prefix=""):
        if self.writer is None:
            return False
        if (self.interval >= 0 and 
                not global_step % self.interval == 0):
            return False
        assert isinstance(outputs, TrainerOutputs)
        for key, value in  outputs.scalar_only().items():
            key = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(key, value, global_step)
        return True
