import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

    # https://numpy.org/neps/nep-0050-scalar-promotion.html
    os.environ['NPY_PROMOTION_STATE'] = 'weak'
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    import numpy as np  # reload numpy with NPY_PROMOTION_STATE='weak'

assert np._get_promotion_state() == 'weak', "numpy import error"

major, minor = np.__version__.split('.')[:2]
if int(major) < 2 and int(minor) < 24:
    raise ImportError(f'NumPy version is too low, requires 1.24 or above')

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


class nparray(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        if input_array.__class__ is cls:
            return input_array
        obj = np.array(input_array, *args, **kwargs).view(cls)
        obj._version = [0]
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        base = self.base
        if base is not None and base.base is not None:  # it's a view of nparray
            self._version = getattr(obj, '_version', [0])
        else:
            self._version = [0]


# cupy
from importlib.util import find_spec


class cparray:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("cupy not installed, can't use cuda")


cupy_is_loaded = bool(find_spec('cupy'))
cp = None
if cupy_is_loaded:
    import cupy as cp

    cp.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    if int(cp.__version__.split('.')[0]) < 10:  # cupy major version lower than 10
        import subprocess

        pip_list = subprocess.Popen(('pip', 'list'), stdout=subprocess.PIPE)
        cupy_name = subprocess.check_output(('grep', 'cupy'), stdin=pip_list.stdout).decode('ascii').split(' ')[0]
        raise ImportError(f'Current CuPy version is too low, update it using "!pip install -U {cupy_name}" and '
                          f'restart the runtime.')


    class cparray(cp.ndarray):
        def __new__(cls, input_array, *args, **kwargs):
            if input_array.__class__ is cls:
                return input_array
            obj = cp.array(input_array, *args, **kwargs).view(cls)
            obj._version = [0]
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            if self.base is not None:  # it's a view of cparray
                self._version = getattr(obj, '_version', [0])
            else:
                self._version = [0]

if __name__ == '__main__':
    x = nparray([[1, 2, 3], [4, 5, 6]])
    x._version[0] += 100
    y = x[0, :2]  # keep version for inplace ops
    print(f'y version: {y._version}')
    z = x + 1  # create new for outplace ops
    print(f'z version: {z._version}')

    x = cparray([[1, 2, 3], [4, 5, 6]])
    x._version[0] += 100
    y = cp.transpose(x, (1, 0))  # keep version for inplace ops
    print(f'y version: {y._version}')
    z = cp.sqrt(x)  # create new for outplace ops
    print(f'z version: {z._version}')
