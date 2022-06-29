import math

from .module import *
from .. import functional as F
from .. import init

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.weight = Parameter(zeros((out_features, in_features)))
        if bias:
            self.bias = Parameter(zeros(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        xp = cp if self.weight.__class__ is cp_ndarray else np
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            self.bias.data = xp.random.uniform(low=-bound, high=bound, size=self.bias.shape).astype(np.float32)

    def extra_repr(self):
        return f'in_feature={self.in_feature}, out_feature={self.out_feature}, bias={self.bias is not None}'

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
