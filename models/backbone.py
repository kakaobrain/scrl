import logging

import torch
import torchvision.models as models
from models.swin_xfmr import build_swin_xformer
import torchvision.ops as ops
from models.heads import TwoLayerLinearHead

log = logging.getLogger('main')


class Backbone(torch.nn.Module):
    def __init__(self, name, proj_head_kwargs, scrl_kwargs, trainable=True):
        super(Backbone, self).__init__()
        assert name in ['resnet50', 'resnet101'] or name.startswith('swin')
        self.name = name
        self.scrl_enabled = scrl_kwargs.enabled
        self.trainable = trainable

        # encoder
        if name.startswith('swin'):
            network = build_swin_xformer(name)
            self.encoder = network
        else:
            network = eval(f"models.{name}")()
            self.encoder = torch.nn.Sequential(*list(network.children())[:-1])
        
        # RoI pooling layer for SCRL
        if self.trainable and self.scrl_enabled:
            roi_out_size = (scrl_kwargs.pool_size, ) * 2
            self.roi_align = ops.RoIAlign(output_size=roi_out_size,
                                          sampling_ratio=scrl_kwargs.sampling_ratio,
                                          spatial_scale=scrl_kwargs.spatial_scale,
                                          aligned=scrl_kwargs.detectron_aligned)

        # projection head
        if self.trainable:
            self.projector = TwoLayerLinearHead(**proj_head_kwargs)

    @classmethod
    def init_from_config(cls, cfg):
        return cls(
            name=cfg.network.name,
            proj_head_kwargs=cfg.network.proj_head,
            scrl_kwargs=cfg.network.scrl,
            trainable=cfg.train.enabled,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, boxes=None, no_projection=False):
        """
        Args:
            x: augmented(randomly cropped)images.
            boxes: boxes coordinates to be pooled.
            no_projection: ignore the projection layer (for evaluation)
        Returns:
            p: after projection / roi_p: RoI-aligned feature after projection
            h: before projection
        """
        if self.name.startswith('resnet'):
            for n, layer in enumerate(self.encoder):
                x = layer(x)
                if n == len(self.encoder) - 2:
                    h_pre_gap = x
            h = x.squeeze()
        elif self.name.startswith('swin'):
            h_pre_gap = self.encoder(x)
            h = h_pre_gap.flatten(-2).mean(-1)

        if not self.trainable or no_projection:
            return h

        if self.scrl_enabled:
            assert boxes is not None
            roi_h = self.roi_align(h_pre_gap, boxes).squeeze()
            roi_p = self.projector(roi_h)
            return roi_p, h
        else:
            p = self.projector(h)
            return p, h
