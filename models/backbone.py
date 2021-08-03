import logging

import torch
import torchvision.models as models
from models.swin_xfmr import build_swin_xformer
import torchvision.ops as ops
from models.heads import MultiLayerNonLinearHead

log = logging.getLogger('main')


class Backbone(torch.nn.Module):
    def __init__(self, net_type, network_kwargs, trainable=True):
        super(Backbone, self).__init__()
        assert net_type in ['online', 'target'], \
            "net_type should be either 'online' or 'target'"
        self.name = network_kwargs.name
        assert self.name in ['resnet50', 'resnet101'] or self.name.startswith('swin'), \
            f"Unexpected network name: {self.name}"
        self.scrl_enabled = network_kwargs.scrl.enabled
        self.trainable = trainable

        # encoder
        if self.name.startswith('swin'):
            network = build_swin_xformer(self.name, net_type, network_kwargs.swin.fix_patch_proj)
            self.encoder = network
        else:
            network = eval(f"models.{self.name}")()
            self.encoder = torch.nn.Sequential(*list(network.children())[:-1])
        
        # RoI pooling layer for SCRL
        if self.trainable and self.scrl_enabled:
            roi_out_size = (network_kwargs.scrl.pool_size, ) * 2
            self.roi_align = ops.RoIAlign(output_size=roi_out_size,
                                          sampling_ratio=network_kwargs.scrl.sampling_ratio,
                                          spatial_scale=network_kwargs.scrl.spatial_scale,
                                          aligned=network_kwargs.scrl.detectron_aligned)

        # projection head
        if self.trainable:
            self.projector = MultiLayerNonLinearHead(**network_kwargs.proj_head)

    @classmethod
    def init_online_from_config(cls, cfg):
        return cls(
            net_type='online',
            network_kwargs=cfg.network,
            trainable=cfg.train.enabled,
        )
        
    @classmethod
    def init_target_from_config(cls, cfg):
        return cls(
            net_type='target',
            network_kwargs=cfg.network,
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
            h = h_pre_gap.flatten(-2).mean(-1)  # GAP

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
