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

import numpy as np

np.set_printoptions(precision=4)

# check modules
from importlib.util import find_spec

cp_ndarray = False
cupy_is_loaded = bool(find_spec('cupy'))
cp = np  # cupy defaults to numpy
if cupy_is_loaded:
    # os.environ['CUPY_ACCELERATORS'] = 'cub'
    import cupy as cp

    if int(cp.__version__.split('.')[0]) < 10:  # cupy major version lower than 10
        import subprocess

        pip_list = subprocess.Popen(('pip', 'list'), stdout=subprocess.PIPE)
        cupy_name = subprocess.check_output(('grep', 'cupy'), stdin=pip_list.stdout).decode('ascii').split(' ')[0]
        raise ImportError(f'Current CuPy version is too low, update it using "!pip install -U {cupy_name}" and '
                          f'restart the runtime.')
    cp_ndarray = cp.ndarray
    cp.set_printoptions(precision=4)

from tortto import _version

__version__ = _version.__version__

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
