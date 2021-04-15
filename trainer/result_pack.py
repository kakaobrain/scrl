from collections import OrderedDict
import logging
import os
import pandas as pd
import seaborn as sns

import torch

log = logging.getLogger('main')


class ResultPack(object):
    def __init__(self, exp_name, columns=None, metadata={}):
        self.exp_name = exp_name.split('/')[-1]  # get basename
        self.metadata = metadata
        if columns is not None:
            self.initialize(columns)
        else:
            self.columns = None
            self.df = None

    def __repr__(self):
        return f"ResultPack: \n{self.df.__repr__()}"

    def __getstate__(self):
        return {
            'exp_name': self.exp_name,
            'columns': self.columns,
            'df': self.df,
            'metadata': self.metadata,
        }

    def __setstate__(self, state):
        self.exp_name = state['exp_name']
        self.columns = state['columns']
        self.df = state['df']
        self.metadata = state['metadata']
        
    def initialize(self, columns):
        self.columns = columns
        # attach exp_name for the case where df is merged from different experiments
        self.df = pd.DataFrame(columns=['exp_name'] + list(columns))

    def new(self, exp_name=None):
        return ResultPack(exp_name or self.exp_name, self.columns)

    def append(self, epoch, **kwargs):
        if self.columns is None:
            assert self.df is None
            self.initialize(tuple(kwargs.keys()))
        else:
            assert set(kwargs.keys()) == set(self.columns), f"expected keys: {self.columns}"
        # convert a tensor to a float type if there is
        args = [kwargs[key].item() 
                if isinstance(kwargs[key], torch.Tensor) else kwargs[key] 
                for key in self.columns]
        self.df.loc[epoch] = [self.exp_name] + args
        return self

    def get_the_latest_row(self, **kwargs):
        ret = self.df.tail(1).rename_axis('epochs').reset_index().to_dict()
        ret.update(kwargs)
        return ret

    def tail(self, n):
        pack_ret = self.new()
        pack_ret.df = self.df.tail(n)
        return pack_ret

    def get_latest_by_key(self, key):
        return self.df.tail(1).rename_axis('epochs').reset_index().to_dict()[key][0]
    
    def get_latest_by_keys(self, keys):
        dict_latest = self.df.tail(1).rename_axis(
            'epochs').reset_index().to_dict()
        dict_ret = OrderedDict()
        try:
            for key in keys:
                dict_ret[key] = dict_latest[key][0]
        except KeyError:
            raise KeyError(
                f"Can't find column '{key}'. Available columns: {self.columns}")
        return dict_ret
    
    def get_latest_columns(self):
        return self.get_latest_by_keys(self.columns)

    def save_as_csv(self, save_dir, name):
        save_dir = os.path.join(save_dir, f"{name}.csv")
        self.df.rename_axis('epochs').to_csv(save_dir)
        log.info(f"Result pack has been saved as a CSV file to: {save_dir}")

    @staticmethod
    def concat(exp_name, result_packs):
        assert isinstance(result_packs, (list, tuple))
        assert isinstance(result_packs[0], ResultPack)
        pack_ret = result_packs[0].new(exp_name=exp_name)
        pack_ret.df = pd.concat([result_pack.df for result_pack in result_packs])
        return pack_ret

    def save_as_plot(self, save_dir, title=""):
        sns.set_theme(style="darkgrid")
        data = self.df.rename_axis('epochs')
        for column in self.columns:
            plot = sns.lineplot(data=data, x="epochs", y=column, hue='exp_name',
                                palette='pastel', legend="full")
            save_dir_ = os.path.join(save_dir, f'{column}.png')
            plot.get_figure().savefig(save_dir_)
            plot.get_figure().clf()
            log.info(f"Result pack has been saved as a figure to: {save_dir_}")
