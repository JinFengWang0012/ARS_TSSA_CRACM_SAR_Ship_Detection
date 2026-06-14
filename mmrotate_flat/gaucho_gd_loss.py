from copy import deepcopy

import torch
from torch import nn

from mmrotate.models.losses.gaussian_dist_loss import (gwd_loss, jd_loss,
                                                       kld_loss,
                                                       kld_symmax_loss,
                                                       kld_symmin_loss)
from mmrotate.models.losses.gaussian_dist_loss_v1 import bcd_loss
from mmrotate.models.utils.gaucho import cholesky_to_gaussian, rbox_to_gaussian
from mmrotate.registry import MODELS


@MODELS.register_module()
class GauChoGDLoss(nn.Module):
    """Gaussian loss wrapper for GauCho Cholesky parameters."""

    BAG_GD_LOSS = {
        'gwd': gwd_loss,
        'kld': kld_loss,
        'jd': jd_loss,
        'kld_symmax': kld_symmax_loss,
        'kld_symmin': kld_symmin_loss,
        'bcd': bcd_loss
    }

    def __init__(self,
                 loss_type='gwd',
                 fun='log1p',
                 tau=1.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        if (weight is not None) and (not torch.any(weight > 0)) and \
                reduction != 'none':
            return pred.sum() * 0

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        extra_kwargs = deepcopy(self.kwargs)
        extra_kwargs.update(kwargs)

        pred = cholesky_to_gaussian(pred)
        target = rbox_to_gaussian(target)

        return self.loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **extra_kwargs) * self.loss_weight
