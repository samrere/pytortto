"""
https://docs.cupy.dev/en/stable/reference/environment.html
https://docs.cupy.dev/en/stable/user_guide/performance.html
import os
os.environ['CUPY_TF32']='1' # only available on GPUs with compute capability 8.0 or higher
os.environ['CUPY_ACCELERATORS']='cub'
"""
# environ parameters
import os

fft_conv = True if os.environ.get("FFT") == 'True' or os.environ.get("fft") == 'True' else False

from .xparray import *

__version__ = '1.3.0'

# integer 0, used to initialize Tensor .grad, and in backward().
# also imported by optim.optimizer, used in zero_grad
# also imported by adam: if p.grad is not _int_zero --> p has gradient
# notice if gradient is 0D numpy array then gradient update: _int_zero += np.array(3, dtype=np.float32) will be float32,
# so set _int_zero to int8
_int_zero = np.int8(0)

from .tensor import *
from .VariableFunctions import *
from .serialization import *
from . import nn
from . import optim
from .autograd.helper import *
from .autograd.grad_fcn import *
from .autograd.grad_ufunc import *
