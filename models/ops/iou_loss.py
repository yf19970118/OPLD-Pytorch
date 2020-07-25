import torch
from torch import nn
import math


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ac_uion = g_w_intersect * g_h_intersect + 1e-7

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'diou' or self.loc_loss_type == 'ciou':
            target_center_x = (target_right - target_left) / 2
            target_center_y = (target_top - target_bottom) / 2
            pred_center_x = (pred_right - pred_left) / 2
            pred_center_y = (pred_top - pred_bottom) / 2

            inter_diag = (target_center_x - pred_center_x) ** 2 + (target_center_y - pred_center_y) ** 2
            outer_diag = g_w_intersect ** 2 + g_h_intersect ** 2
            u = inter_diag / outer_diag
            dious = ious - u

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'liou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        elif self.loc_loss_type == 'diou':
            losses = 1 - dious
        elif self.loc_loss_type == 'ciou':
            v = (4 / (math.pi ** 2)) * torch.pow((
                    torch.atan((target_left + target_right) / (target_top + target_bottom + 1e-7)) -
                    torch.atan((pred_left + pred_right) / (pred_top + pred_bottom + 1e-7))), 2)
            S = 1 - ious
            alpha = v / (S + v)
            cious = ious - (u + alpha * v)
            losses = 1 - cious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()
