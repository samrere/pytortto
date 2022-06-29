from .module import *
from .utils import _pair
from .. import functional as F


class _MaxPoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool2d(_MaxPoolNd):

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(input, _pair(self.kernel_size), _pair(self.stride),
                            _pair(self.padding), _pair(self.dilation), self.ceil_mode,
                            self.return_indices)
