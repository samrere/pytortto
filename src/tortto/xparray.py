import numpy as np
np.set_printoptions(precision=4)
class nparray(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.array(input_array, *args, **kwargs).view(cls)
        obj._version = 0
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self._version = getattr(obj, '_version', 0)


# cupy
from importlib.util import find_spec
cparray = None
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

    class cparray(cp.ndarray):
        def __new__(cls, input_array, *args, **kwargs):
            obj = cp.array(input_array, *args, **kwargs).view(cls)
            obj._version = 0

            return obj
        def __array_finalize__(self, obj):
            if obj is None: return
            self._version = getattr(obj, '_version', 0)


    cp.set_printoptions(precision=4)