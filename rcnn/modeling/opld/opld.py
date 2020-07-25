import torch
from torch import nn

from rcnn.modeling.opld import heads
from rcnn.modeling.opld import outputs
from rcnn.modeling.opld.inference import post_processor
from rcnn.modeling.opld.loss import loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class OPLD(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(OPLD, self).__init__()
        head = registry.ROI_OPLD_HEADS[cfg.OPLD.ROI_OPLD_HEAD]
        self.Head = head(dim_in, spatial_scale)
        output = registry.ROI_OPLD_OUTPUTS[cfg.OPLD.ROI_OPLD_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = post_processor()
        self.loss_evaluator = loss_evaluator()

    def forward(self, features, proposals, targets=None):
        if self.training:
            x, result, loss = self._forward_train_grid(features, proposals, targets)
            return x, result, loss
        else:
            x, result, _ = self._forward_test_grid(features, proposals)
            return x, result, {}

    def _forward_train_grid(self, features, proposals, targets=None):
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.resample(proposals, targets)

        x, x_so = self.Head(features, proposals)
        grid_logits = self.Output(x, x_so)

        loss_grid = self.loss_evaluator(grid_logits)

        return x, all_proposals, dict(loss_grid=loss_grid)

    def _forward_test_grid(self, features, proposals):
        x, x_so = self.Head(features, proposals)
        grid_logits = self.Output(x, x_so)

        result = self.post_processor(grid_logits, proposals)
        return x, result, {}
