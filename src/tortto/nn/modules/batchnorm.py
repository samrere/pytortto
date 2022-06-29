from tortto import *
from .module import Module
from .. import functional as F
from .. import init
from ..parameter import Parameter


class _NormBase(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # check out doc for batchnorm1d:
            # 1xL for 2d and 1xC for 3d, need to append a new dim if 3d --> 1xCx1 don't forget to
            # do it during forward and backward. can't do it here because we don't know data dim
            # during init
            self.weight = Parameter(ones(num_features))
            self.bias = Parameter(zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', zeros(num_features))
            self.register_buffer('running_var', ones(num_features))
            self.register_buffer('num_batches_tracked', Tensor(0))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean = init.zeros_(self.running_mean)
            self.running_var = init.ones_(self.running_var)
            self.num_batches_tracked = init.zeros_(self.num_batches_tracked)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, inpt: Tensor) -> Tensor:
        self._check_input_dim(inpt)

        if self.training and self.track_running_stats:
            self.num_batches_tracked.data += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        else:
            exponential_average_factor = None

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # move bn_training location, so no default value for training
        return F.batch_norm(inpt,
                            # If buffers are not to be tracked, ensure that they won't be updated
                            self.running_mean if not self.training or self.track_running_stats else None,
                            self.running_var if not self.training or self.track_running_stats else None,
                            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))


if __name__ == '__main__':
    ...
