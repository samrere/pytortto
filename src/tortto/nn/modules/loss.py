from .module import *
from .. import functional as F


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inpt, target):
        return F.mse_loss(inpt, target, reduction=self.reduction)


class BCELoss(Module):
    def __init__(self, reduction='mean', weight=None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction='mean', weight=None, pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target,
                                                  weight=self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class NLLLoss(Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inpt, target):
        return F.nll_loss(inpt, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
