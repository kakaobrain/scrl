import collections
from collections import OrderedDict
from copy import deepcopy
import logging
from os.path import basename, splitext
from pprint import pformat
from types import SimpleNamespace
import yaml

import torch

from .color import Colorer
from .logger import set_file_handler, set_stream_handler
from .utils import turn_into_debug_config, get_auto_save_dir, create_save_dir
from distributed import comm, get_dist_url

log = logging.getLogger('main')
C = Colorer.instance()


def get_cfg(args):        
    # load config from yaml and overwrite parsed args
    cfg = Config.from_yaml(args.config)
    cfg.update(vars(args))  # overwrite .yaml values with parsed_args

    # set singleton colorizer
    C.set_enabled(not cfg.no_color)

    # loggers
    set_stream_handler('main', cfg.log_level)
    log.info(C.green(f"[!] Initializing configuration.."))
    
    # set runtime configuration
    url_origin = cfg.dist_url
    cfg.dist_url = get_dist_url(cfg.dist_url, cfg.num_machines)
    cfg.config_name = splitext(basename(cfg.config))[0]
    if cfg.debug:
        cfg = turn_into_debug_config(cfg)
    if cfg.num_gpus <= 0:
        cfg.num_gpus = torch.cuda.device_count()
    if not cfg.save_dir:
        # make a suffixed directory with automatic numbering
        cfg.save_dir = get_auto_save_dir(config_name=cfg.config_name,
                                         top_dir=cfg.top_dir,
                                         overwrite=cfg.overwrite)
    # print configuration
    log.info("[CFG] " + C.cyan(f'\n{cfg.pformat()}'))
    create_save_dir(save_dir=cfg.save_dir, backup_files=[cfg.config])
        
    # empahsize primary fields
    if cfg.debug:
        log.debug(f"[CFG] Debugging mode is on!")
    url_origin = f"{url_origin} -> " if cfg.dist_url != url_origin else ""
    log.info(f"[CFG] dist_url: {url_origin}{cfg.dist_url}")
    log.info(f"[CFG] config from: {cfg.config}")
    log.info(f"[CFG] network: {cfg.network.name}")
    log.info(f"[CFG] method: {'SCRL' if cfg.network.scrl.enabled else 'BYOL'}")
    log.info(f"[CFG] train.enabled: {'True' if cfg.train.enabled else 'False'} / "
             f"eval.enabled: {'True' if cfg.eval.enabled else 'False'}")  
    if cfg.train.enabled:
        log.info(f"[CFG] train/num_works: {cfg.train.num_workers}")
    if cfg.eval.enabled:
        log.info(f"[CFG] eval/num_works: {cfg.eval.num_workers}")     
    return cfg 


def _update_dict(tar, src):
    """recursive dict update."""
    for k, v in src.items():
        if isinstance(v, collections.abc.Mapping):
            tar[k] = _update_dict(tar.get(k, {}), v)
        else:
            tar[k] = v
    return tar


class Config(SimpleNamespace):
    """Dictionary-based but also dot-accessible configuration object, which will 
    rescue you from the messy brackets and quotation marks while accessing 
    nested dictionaries.
        
    As the usage example below, a value can be easily assigned to a new field 
    with hierarchies by using Python's usual assignment syntax. Due to the side 
    effects of this feature, it is safe that the user call '.freeze()' before 
    using the Config instance as a fixed configuration. Otherwise, even when 
    a wanted attribute is called with an incorrect name, AttributeError will be 
    silently ignored and returns an empty config, which could be resulting in 
    unwanted consequences.
    
    Usage:
        >>> cfg = Config()
        >>> cfg.foo = 1
        >>> cfg.bar.baz = 2
        >>> cfg['bar']['baz'] == cfg.bar.baz
        True
        >>> cfg.pprint()
        ---
        foo: 1
        bar:
            baz: 2
        ...
        >>> cfg.freeze()
        >>> cfg.new = 3
        RuntimeError: Can't set new attribute after being freezed!
            
    """
    def __init__(self, _dict=None, **kwargs):
        super().__init__(**kwargs)
        self._freezed = False
        self._order = list()
        if _dict is not None:
            self._set_with_nested_dict(_dict)

    def _set_with_nested_dict(self, _dict):
        for key, value in _dict.items():
            if isinstance(value, dict):
                self.__setattr__(key, Config(value))
            else:
                self.__setattr__(key, value)
                self._order.append(key)
                
    @property
    def freezed(self):
        return self._freezed
                
    @classmethod
    def from_yaml(cls, yaml_file):
        """Initialize configuration with a YAML file."""
        return cls(OrderedDict(yaml.load(open(yaml_file, "r"), 
                                         Loader=yaml.FullLoader)))

    def __repr__(self):
        return 'Config' + self.to_dict().__repr__()

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError as e:
            if self._freezed:
                raise AttributeError(f"Can't find the field: {item}") from e
            else:
                # if there's no attribute with the given name, 
                # make new one and assign an empty config. 
                self.__setattr__(item, Config())
                return self.__getattribute__(item)
        
    def __setattr__(self, item, value):
        if item != '_freezed' and self.__dict__['_freezed']:
            raise RuntimeError("Can't set new attribute after being freezed!")
        super().__setattr__(item, value)

    def __bool__(self):
        return len([k for k in self.to_dict().keys() 
                    if not k.startswith('_')]) > 0

    def __len__(self):
        return len(self.to_dict())

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self._set_with_nested_dict(state)

    def __contains__(self, item):
        return self.to_dict().__contains__(item)

    def __deepcopy__(self, memodict={}):
        return Config(_dict=deepcopy(self.to_dict()))

    def __iter__(self):
        # for iterable unpacking
        return self.to_dict().__iter__()
    
    def pformat(self):
        return yaml.dump(self.to_dict(), indent=4, sort_keys=False,
                         explicit_start=True, explicit_end=True)
                                        
    def pprint(self):
        return print(self.pformat())
    
    def freeze(self):
        self._freezed = True
        for value in self.__dict__.values():
            if isinstance(value, Config):
                value.freeze()
        
        return self
        
    def defrost(self):
        self._freezed = False
        for value in self.__dict__.values():
            if isinstance(value, Config):
                value.defrost()
        return self

    def get(self, *args, **kwargs):
        return self.to_dict().get(*args, **kwargs)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def clone(self):
        return self.__deepcopy__()

    def update(self, dict_, delimiter='/'):
        for k, v in dict_.items():
            self._update(k, v, delimiter)

    def _update(self, key, value, delimiter='/'):
        obj = self
        keys = key.split(delimiter)
        for k in keys[:-1]:
            obj = obj.__getattr__(k)
        obj.__setattr__(keys[-1], value)

    def to_dict(self):
        out_dict = OrderedDict()
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out_dict[key] = value.to_dict()
            else:
                if not key.startswith('_'):
                    out_dict[key] = value
        return dict(out_dict)
