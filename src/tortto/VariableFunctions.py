import warnings
from typing import Union, Tuple

import tortto as tt
from tortto import np, cp, cp_ndarray


def manual_seed(seed):
    np.random.seed(seed)
    cp.random.seed(seed)


# def to_array(arr):
#     if cp.__name__=='cupy':
#         return cp.array(arr)
#     else: # if uses numpy
#         if not isinstance(arr,np_ndarray): # if incoming array is cupy array
#             return arr.get()
#     return arr

def _values_like(fcn, tensor, **kwargs):
    dtype = kwargs.get('dtype')
    return tt.Tensor(fcn(tensor.data, dtype=dtype), dtype=dtype, copy=False, **kwargs)


def _values(fcn, shape, dtype=np.float32, **kwargs):
    # if not isinstance(shape, tuple):
    #     raise TypeError('shape should be tuple')
    return tt.Tensor(fcn(shape, dtype=dtype), dtype=dtype, copy=False, **kwargs)


def randn_like(tensor, **kwargs):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    return tt.Tensor(xp.random.randn(*tensor.shape), dtype=tensor.dtype, copy=False, **kwargs)


def empty(shape: Union[int, Tuple], **kwargs):
    return _values(np.empty, shape, **kwargs)


def zeros(shape: Union[int, Tuple], **kwargs):
    return _values(np.zeros, shape, **kwargs)


def ones(shape: Union[int, Tuple], **kwargs):
    return _values(np.ones, shape, **kwargs)


def empty_like(tensor, **kwargs):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    return _values_like(xp.empty_like, tensor, **kwargs)


def zeros_like(tensor, **kwargs):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    return _values_like(xp.zeros_like, tensor, **kwargs)


def ones_like(tensor, **kwargs):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    return _values_like(xp.ones_like, tensor, **kwargs)


def randn(*shape, **kwargs):
    if hasattr(shape[0], '__iter__'):
        shape = shape[0]
    output = tt.Tensor(np.random.randn(*shape), **kwargs)
    return output


def eye(N, M=None, k=0):
    return tt.Tensor(np.eye(N, M, k))


def arange(start, end=None, step=1, **kwargs):
    if end is None:
        start, end = 0, start
    return tt.Tensor(np.arange(start, end, step), **kwargs)


def linspace(start, end, steps, **kwargs):
    return tt.Tensor([np.linspace(start, end, steps)], **kwargs)


def tensor(data, requires_grad=False, dtype=None, copy=True):
    if isinstance(data, tt.Tensor):
        warnings.warn('Input is already a tensor, returning without deepcopy', UserWarning)
    else:
        if dtype is None:
            dtype = getattr(data, 'dtype', tt.float32)
        return tt.Tensor(data, requires_grad=requires_grad, dtype=dtype, copy=copy)


def flatten(x, start_dim=0, end_dim=-1):
    # convert neg indices to positive, making end_dim inclusive
    shape = x.shape
    if end_dim < 0:
        end_dim = len(shape) + end_dim
    end_dim += 1  # inclusive
    if start_dim < 0:
        start_dim = len(shape) + start_dim  # exclusive
    if start_dim >= end_dim:
        raise RuntimeError('flatten() has invalid args: start_dim cannot come after end_dim')
    # construct newshape
    newshape = shape[:start_dim]
    if start_dim != end_dim:
        newshape += (-1,)
    if end_dim < len(shape):
        newshape += shape[end_dim:]
    if len(newshape) != len(shape):
        return x.view(newshape)
    else:
        return x


def logical_or(x1, x2):
    xp = cp if x1.data.__class__ is cp_ndarray else np
    return tt.tensor(xp.logical_or(x1.data, x2.data))


def logical_and(x1, x2):
    xp = cp if x1.data.__class__ is cp_ndarray else np
    return tt.tensor(xp.logical_and(x1.data, x2.data))


def logical_not(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return tt.tensor(xp.logical_not(x.data))

def logical_xor(x1, x2):
    xp = cp if x1.data.__class__ is cp_ndarray else np
    return tt.tensor(xp.logical_xor(x1.data, x2.data))

def argmax(x, dim=None, keepdim=False):
    # numpy version lower than 1.22.0. don't have the `keepdims` keyword.
    if keepdim:
        return tt.tensor(x.data.argmax(axis=dim, keepdims=True), dtype=tt.int64)
    else:
        return tt.tensor(x.data.argmax(axis=dim), dtype=tt.int64)

def argmin(x, dim=None, keepdim=False):
    if keepdim:
        return tt.tensor(x.data.argmin(axis=dim, keepdims=True), dtype=tt.int64)
    else:
        return tt.tensor(x.data.argmin(axis=dim), dtype=tt.int64)
