import torch
from torch import nn


class SingleLayerLinearHead(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerLinearHead, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    @classmethod
    def init_evaluator_from_config(cls, cfg, num_classes):
        return cls(
            input_size=cfg.network.proj_head.input_size,
            output_size=num_classes,
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class TwoLayerLinearHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLinearHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size, bias=False)
            # suppress bias for the last layer as in the BYOL official code
        )
        
    @classmethod
    def init_projector_from_config(cls, cfg):
        return cls(
            input_size=cfg.network.proj_head.input_size,
            hidden_size=cfg.network.proj_head.hidden_size,
            output_size=cfg.network.proj_head.output_size,
        )
        
    @classmethod
    def init_predictor_from_config(cls, cfg):
        return cls(
            input_size=cfg.network.proj_head.output_size,
            hidden_size=cfg.network.proj_head.hidden_size,
            output_size=cfg.network.proj_head.output_size,
        )

    def forward(self, x):
        return self.net(x)
