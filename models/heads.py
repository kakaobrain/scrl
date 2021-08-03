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


class MultiLayerNonLinearHead(nn.Module):
    def __init__(
        self, num_layers, input_size, hidden_size, output_size, output_bn):
        super(MultiLayerNonLinearHead, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            _input_size = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(_input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Linear(hidden_size, output_size, bias=False))
        if output_bn:
            layers.append(nn.BatchNorm1d(output_size))

        self.net = nn.Sequential(*layers)
        
    @classmethod
    def init_projector_from_config(cls, cfg):
        return cls(
            num_layers=cfg.network.proj_head.num_layers,
            input_size=cfg.network.proj_head.input_size,
            hidden_size=cfg.network.proj_head.hidden_size,
            output_size=cfg.network.proj_head.output_size,
            output_bn=cfg.network.proj_head.output_bn,
        )
        
    @classmethod
    def init_predictor_from_config(cls, cfg):
        return cls(
            num_layers=2,
            input_size=cfg.network.proj_head.output_size,
            hidden_size=cfg.network.proj_head.hidden_size,
            output_size=cfg.network.proj_head.output_size,
            output_bn=cfg.network.proj_head.output_bn,
        )

    def forward(self, x):
        return self.net(x)
