import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from utils.net import make_conv
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_OPLD_HEADS.register("roi_opld_head")
class RoIOPLDHead(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(RoIOPLDHead, self).__init__()
        self.num_points = cfg.OPLD.NUM_POINTS
        self.roi_feat_size = cfg.OPLD.ROI_FEAT_SIZE

        self.num_convs = cfg.OPLD.ROI_HEAD.NUM_CONVS
        self.point_feat_channels = cfg.OPLD.ROI_HEAD.POINT_FEAT_CHANNELS
        self.neighbor_points = cfg.OPLD.ROI_HEAD.NEIGHBOR_POINTS

        self.conv_out_channels = self.point_feat_channels * self.num_points
        self.class_agnostic = False
        self.dim_in = dim_in[-1]

        self.whole_map_size = self.roi_feat_size * 4
        self.convs = []
        conv_kernel_size = 3
        stride = 1
        for i in range(self.num_convs):
            in_channels = (self.dim_in if i == 0 else self.conv_out_channels)
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.conv_out_channels, kernel_size=conv_kernel_size, stride=stride,
                              padding=padding),
                    nn.GroupNorm(32, self.conv_out_channels, eps=1e-5),
                    nn.ReLU(inplace=True)
                )
            )
        self.convs = nn.Sequential(*self.convs)

        self.forder_trans = self._build_trans(nn.ModuleList())  # first-order feature transition
        self.sorder_trans = self._build_trans(nn.ModuleList())  # second-order feature transition

        method = cfg.OPLD.ROI_XFORM_METHOD
        resolution = cfg.OPLD.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.OPLD.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        self.dim_out = [self.conv_out_channels]

    def _build_trans(self, trans):
        for neighbors in self.neighbor_points:
            _trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                # each transition module consists of a 5x5 depth-wise conv and a 1x1 conv.
                _trans.append(
                    nn.Sequential(
                        nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 5, stride=1, padding=2,
                                  groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 1))
                )
            trans.append(_trans)
        return trans

    def forward(self, features, proposals):
        x = self.pooler(features, proposals)
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        # RoI feature transformation, get 14*14
        x = self.convs(x)

        c = self.point_feat_channels
        # first-order fusion
        x_fo = [None for _ in range(self.num_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](x[:, point_idx * c:(point_idx + 1) * c])

        # second-order fusion
        x_so = [None for _ in range(self.num_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])

        return x, x_so
