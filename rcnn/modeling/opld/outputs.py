import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from models.ops import Conv2d
from models.ops import ConvTranspose2d
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_OPLD_OUTPUTS.register("opld_output")
class OPLDOutput(nn.Module):
    def __init__(self, dim_in):
        super(OPLDOutput, self).__init__()
        self.dim_in = dim_in[-1]
        self.num_points = cfg.OPLD.NUM_POINTS
        self.point_feat_channels = cfg.OPLD.ROI_HEAD.POINT_FEAT_CHANNELS
        self.conv_out_channels = self.point_feat_channels * self.num_points
        deconv_kernel_size = 4

        self.norm1 = nn.GroupNorm(self.num_points, self.conv_out_channels)
        self.deconv_1 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.num_points)
        self.deconv_2 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.num_points,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.num_points)

    def forward(self, x, x_so):
        # predicted heatmap with fused features
        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv_1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv_2(x2)
        # predicted heatmap with original features (applicable during training)
        if self.training:
            x1 = x
            x1 = self.deconv_1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv_2(x1)
        else:
            heatmap_unfused = heatmap

        return dict(fused=heatmap, unfused=heatmap_unfused)
