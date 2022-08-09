from .module import Module
from .. import functional as F


class _DropoutNd(Module):
    def __init__(self, p=0.5, inplace=False):
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout(_DropoutNd):
    def forward(self, inpt):
        return F.dropout(inpt, self.p, self.training, self.inplace)
