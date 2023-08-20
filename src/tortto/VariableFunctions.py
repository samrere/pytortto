import warnings

from tortto import np, cp, cparray
from .autograd.grad_ufunc import *
from .autograd.grad_fcn import *

def manual_seed(seed):
    np.random.seed(seed)

def empty_like(tensor, dtype=None, requires_grad=False):
    xd0 = tensor.data
    if dtype is None:
        dtype = xd0.dtype
    xp = cp if xd0.__class__ is cparray else np
    return tt.Tensor(xp.empty_like(xd0, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def zeros_like(tensor, dtype=None, requires_grad=False):
    xd0 = tensor.data
    if dtype is None:
        dtype = xd0.dtype
    xp = cp if xd0.__class__ is cparray else np
    return tt.Tensor(xp.zeros_like(xd0, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def ones_like(tensor, dtype=None, requires_grad=False):
    xd0 = tensor.data
    if dtype is None:
        dtype = xd0.dtype
    xp = cp if xd0.__class__ is cparray else np
    return tt.Tensor(xp.ones_like(xd0, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def randn_like(tensor, dtype=None, requires_grad=False):
    xd0 = tensor.data
    if dtype is None:
        dtype = xd0.dtype
    xp = cp if xd0.__class__ is cparray else np
    return tt.Tensor(xp.random.randn(*xd0.shape), dtype=dtype, copy=False, requires_grad=requires_grad)


def empty(*shape, dtype=None, requires_grad=False):
    if hasattr(shape[0], '__iter__'):
        shape = shape[0]
    if dtype is None:
        dtype = tt.float32
    return tt.Tensor(np.empty(shape, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def zeros(*shape, dtype=None, requires_grad=False):
    if hasattr(shape[0], '__iter__'):
        shape = shape[0]
    if dtype is None:
        dtype = tt.float32
    return tt.Tensor(np.zeros(shape, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def ones(*shape, dtype=None, requires_grad=False):
    if hasattr(shape[0], '__iter__'):
        shape = shape[0]
    if dtype is None:
        dtype = tt.float32
    return tt.Tensor(np.ones(shape, dtype=dtype), dtype=dtype, copy=False, requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    if hasattr(shape[0], '__iter__'):
        shape = shape[0]
    output = tt.Tensor(np.random.randn(*shape), dtype=dtype, copy=False, requires_grad=requires_grad)
    return output


def eye(N, M=None, k=0, **kwargs):
    return tt.Tensor(np.eye(N, M, k), **kwargs)


def arange(start, end=None, step=1, **kwargs):
    if end is None:
        start, end = 0, start
    return tt.Tensor(np.arange(start, end, step), **kwargs)


def linspace(start, end, steps, **kwargs):
    return tt.Tensor([np.linspace(start, end, steps)], **kwargs)


def tensor(data, requires_grad=False, dtype=None, copy=True, **kwargs):

    if isinstance(data, tt.Tensor):
        warnings.warn('Input is already a tensor, returning without deepcopy', UserWarning)
    else:
        if dtype is None:
            dtype = getattr(data, 'dtype', tt.float32)
        if 'grad_fn' in kwargs and not requires_grad:
            kwargs['grad_fn'] = None
        ## if data is not nparray nor cparray, _version will be reset to [0]
        result = tt.Tensor(data, requires_grad=requires_grad, dtype=dtype, copy=copy, **kwargs)
        return result


def flatten(x, start_dim=0, end_dim=-1):
    shape = x.shape
    # convert neg indices to positive, making end_dim inclusive
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
    xp = cp if x1.data.__class__ is cparray else np
    return tt.tensor(xp.logical_or(x1.data, x2.data))


def logical_and(x1, x2):
    xp = cp if x1.data.__class__ is cparray else np
    return tt.tensor(xp.logical_and(x1.data, x2.data))


def logical_not(x):
    xp = cp if x.data.__class__ is cparray else np
    return tt.tensor(xp.logical_not(x.data))


def logical_xor(x1, x2):
    xp = cp if x1.data.__class__ is cparray else np
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


"""
start: from grad_ufunc
"""
def sqrt(input):
    return Sqrt.apply(input, inplace=False)


def sqrt_(input):
    return Sqrt.apply(input, inplace=True)


def exp(input):
    return Exp.apply(input, inplace=False)


def exp_(input):
    return Exp.apply(input, inplace=True)


def tan(input):
    return Tan.apply(input, inplace=False)


def tan_(input):
    return Tan.apply(input, inplace=True)


def tanh(input):
    return Tanh.apply(input, inplace=False)


def tanh_(input):
    return Tanh.apply(input, inplace=True)


def sigmoid(input):
    return Sigmoid.apply(input, inplace=False)


def sigmoid_(input):
    return Sigmoid.apply(input, inplace=True)


def sign(input):
    return Sign.apply(input)


def neg(input):
    return Neg.apply(input, inplace=False)


negative = neg


def neg_(input):
    return Neg.apply(input, inplace=True)


negative_ = neg_


def add(input, other):
    return Add.apply(input, other, inplace=False)


def sub(input, other):
    return Sub.apply(input, other, inplace=False)


subtract = sub


def sin(input):
    return Sin.apply(input, inplace=False)


def sin_(input):
    return Sin.apply(input, inplace=True)


def cos(input):
    return Cos.apply(input, inplace=False)


def cos_(input):
    return Cos.apply(input, inplace=True)


def log(input):
    return Log.apply(input, inplace=False)


def log_(input):
    return Log.apply(input, inplace=True)


def abs(input):
    return Abs.apply(input, inplace=False)


absolute = abs


def abs_(input):
    return Abs.apply(input, inplace=True)


def pow(input, other):
    return Pow.apply(input, other, inplace=False)


def mul(input, other):
    return Mul.apply(input, other, inplace=False)


multiply = mul


def div(input, other):
    return Div.apply(input, other, inplace=False)


divide = div

def clamp(input, min=None, max=None):
    return Clamp.apply(input, min=min, max=max, inplace=False)

def clamp_(input, min=None, max=None):
    return Clamp.apply(input, min=min, max=max, inplace=True)

def max(input):
    return Max.apply(input)

def min(input):
    return Min.apply(input)

def permute(input, dims):
    return Permute.apply(input, dims=dims)
def moveaxis(input, source, destination):
    ndim = input.ndim
    if not -ndim <= source <= (ndim - 1):
        raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {source})')
    if not -ndim <= destination <= (ndim - 1):
        raise IndexError(
            f'Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {destination})')
    if destination < 0:
        destination += ndim
    dims = list(range(ndim))
    dims.insert(destination, dims.pop(source))
    return Permute.apply(input, dims=dims)

def transpose(input, dim0, dim1):
    return Transpose.apply(input, dim0=dim0, dim1=dim1)
swapaxes=transpose
swapdims=transpose
"""
end: from grad_ufunc
"""


"""
start: from grad_fcn
"""
def mm(input, mat2):
    return Mm.apply(input, mat2)
def mv(input, vec):
    return Mv.apply(input, vec)

def bmm(input, mat1):
    return Bmm.apply(input, mat1)
def matmul(input, other):
    """
    use Mm when possible, it's faster to matmul 2D matrices
    if other is a 2D matrix and input is a nD tensor, collapse all leading dimensions so that input can be 2D
    """
    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError(
            f'both arguments to matmul need to be at least 1D, but they are {input.ndim}D and {other.ndim}D')
    if other.ndim==2:
        if input.ndim==2:
            return Mm.apply(input,other)
        elif input.ndim==1:
            return Mm.apply(input[None],other)[0]
        else:
            leading_dims=input.shape[:-1]
            collapse_leading = View.apply(input, shape=(-1, input.shape[-1]))
            result = Mm.apply(collapse_leading, other)
            return View.apply(result, shape=(*leading_dims, result.shape[-1]))
    if input.ndim==2 and other.ndim==1:
        return Mv.apply(input,other)
    return Bmm.apply(input,other)
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return Addmm.apply(input, mat1, mat2, beta=beta, alpha=alpha)
def sum(input, dim=None, keepdim=False):
    return Sum.apply(input, dim=dim, keepdim=keepdim)
def mean(input, dim=None, keepdim=False):
    return Mean.apply(input, dim=dim, keepdim=keepdim)
def var(input, dim=None, unbiased=True,keepdim=False):
    return Var.apply(input, dim=dim, unbiased=unbiased, keepdim=keepdim)
def cat(tensors, dim=0):
    return Cat.apply(*tensors, dim=dim)
def split(tensor, split_size_or_sections, dim=0):
    return Split.apply(tensor, split_size_or_sections=split_size_or_sections, dim=dim)
def chunk(input, chunks, dim=0):
    dim_size = input.shape[dim]
    if dim<0:
        dim+=input.ndim
    chunk_size = dim_size // chunks + (dim_size % chunks != 0)
    return Split.apply(input, split_size_or_sections=chunk_size, dim=dim)
def squeeze(input, dim=None):
    return Squeeze.apply(input, dim=dim)
def unsqueeze(input, dim):
    return Unsqueeze.apply(input, dim=dim)
def masked_fill(input, mask, value):
    if value.__class__ is not tt.Tensor:
        value=tt.tensor(value, copy=False)
    return MaskedFill.apply(input, value, mask=mask, inplace=False)
def masked_fill_(input, mask, value):
    if value.__class__ is not tt.Tensor:
        value=tt.tensor(value, copy=False)
    return MaskedFill.apply(input, value, mask=mask, inplace=True)
"""
end: from grad_fcn
"""





