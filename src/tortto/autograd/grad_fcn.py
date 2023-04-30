from tortto import np, cp, nparray, cparray, cupy_is_loaded
from .function import *
from .helper import *

buildin_sum = sum

class Permute(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xp.transpose(xd0, params['dims']), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dims=ctx.params['axes']
        xp = cp if gd0.__class__ is cparray else np
        grad0=xp.transpose(gd0, axes=np.argsort(dims))
        return grad0
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

class Transpose(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        dim0, dim1 = params['dim0'], params['dim1']
        yt0 = tt.tensor(xp.swapaxes(xd0, dim0,dim1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim0, dim1=ctx.params['dim0'],ctx.params['dim1']
        xp = cp if gd0.__class__ is cparray else np
        grad0=xp.swapaxes(gd0, dim0,dim1)
        return grad0
def transpose(input, dim0, dim1):
    return Transpose.apply(input, dim0=dim0, dim1=dim1)
swapaxes=transpose
swapdims=transpose


class Mm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        if xd0.ndim == 0 or xd1.ndim == 0:
            raise RuntimeError(
                f'both arguments to matmul need to be at least 1D, but they are {xd0.ndim}D and {xd1.ndim}D')
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        x0_true_shape = (1,) + xd0.shape if xd0.ndim < 2 else xd0.shape
        x1_true_shape = xd1.shape + (1,) if xd1.ndim < 2 else xd1.shape
        if x0_true_shape[-1] != x1_true_shape[-2]:
            raise ValueError(f'Dimension mismatch: input0 has a shape of {xd0.shape} and '
                             f'input1 has a shape of {xd1.shape}')
        yt0 = tt.tensor(xp.matmul(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if xd1.ndim<2:
                xd1=xd1[:,None]
            grad0= reverse_broadcast(gd0 @ xd1.swapaxes(-1, -2), xd0.shape)
        else:
            grad0 = None
        if ctx.needs_input_grad[1]:
            if xd0.ndim<2:
                xd0=xd0[None]
            grad1= reverse_broadcast(xd0.swapaxes(-1, -2) @ gd0, xd1.shape)
        else:
            grad1 = None
        return grad0, grad1
def matmul(input, other):
    return Mm.apply(input, other)
################################################################
class Addmm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs
        xd0, xd1, xd2 = xt0.data, xt1.data, xt2.data
        if xd0.ndim == 0 or xd1.ndim == 0:
            raise RuntimeError(
                f'both arguments to matmul need to be at least 1D, but they are {xd0.ndim}D and {xd1.ndim}D')
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        x0_true_shape = (1,) + xd0.shape if xd0.ndim < 2 else xd0.shape
        x1_true_shape = xd1.shape + (1,) if xd1.ndim < 2 else xd1.shape
        if x0_true_shape[-1] != x1_true_shape[-2]:
            raise ValueError(f'Dimension mismatch: input0 has a shape of {xd0.shape} and '
                             f'input1 has a shape of {xd1.shape}')
        yt0 = tt.tensor(xp.matmul(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if xd1.ndim<2:
                xd1=xd1[:,None]
            grad0= reverse_broadcast(gd0 @ xd1.swapaxes(-1, -2), xd0.shape)
        else:
            grad0 = None
        if ctx.needs_input_grad[1]:
            if xd0.ndim<2:
                xd0=xd0[None]
            grad1= reverse_broadcast(xd0.swapaxes(-1, -2) @ gd0, xd1.shape)
        else:
            grad1 = None
        return grad0, grad1
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return Addmm.apply(input, mat1, mat2, beta=beta, alpha=alpha)
##################################################################

class Sum(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        dim=params['dim']
        keepdim=params['keepdim']
        yt0 = tt.tensor(xp.sum(xd0, axis=dim, keepdims=keepdim), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.params['shape']=xd0.shape
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = cp if gd0.__class__ is cparray else np
        if dim is None:
            grad0=xp.lib.stride_tricks.as_strided(gd0, shape=xd0_shape, strides=[0 for _ in xd0_shape])
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                gd0 = xp.expand_dims(gd0, dim)
            strides = list(gd0.strides)
            for i in dim:
                strides[i] = 0  # repeat along axis in x.shape
            grad0=xp.lib.stride_tricks.as_strided(gd0, shape=xd0_shape, strides=strides)
        grad0 = cparray(grad0) if gd0.__class__ is cparray else nparray(grad0)
        return grad0
def sum(input, dim=None, keepdim=False):
    return Sum.apply(input, dim=dim, keepdim=keepdim)


class Mean(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        dim=params['dim']
        keepdim=params['keepdim']
        yt0 = tt.tensor(xp.mean(xd0, axis=dim, keepdims=keepdim), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = cp if gd0.__class__ is cparray else np
        if dim is None:
            grad0=xp.lib.stride_tricks.as_strided(xp.divide(gd0, xd0.size, dtype=gd0.dtype), shape=xd0.shape, strides=[0 for _ in xd0.shape])
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                gd0 = xp.expand_dims(gd0, dim)
            N = 1
            strides = list(gd0.strides)
            for i in dim:
                N *= xd0.shape[i]
                strides[i] = 0  # repeat along axis in x.shape
            grad0=xp.lib.stride_tricks.as_strided(xp.divide(gd0, N, dtype=gd0.dtype), shape=xd0.shape, strides=strides)
        grad0 = cparray(grad0) if gd0.__class__ is cparray else nparray(grad0)
        return grad0
def mean(input, dim=None, keepdim=False):
    return Mean.apply(input, dim=dim, keepdim=keepdim)


class Var(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        dim=params['dim']
        unbiased=params['unbiased']
        keepdim=params['keepdim']
        yt0 = tt.tensor(xp.var(xd0, axis=dim, ddof=unbiased, keepdims=keepdim), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        dim = ctx.params['dim']
        unbiased = ctx.params['unbiased']
        keepdim = ctx.params['keepdim']
        xp = cp if gd0.__class__ is cparray else np
        if not keepdim:
            gd0 = xp.expand_dims(gd0, dim)
        me = xp.mean(xd0, axis=dim, keepdims=True)
        if dim is None:  # if axis is None
            N = xd0.size  # all element
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            N = 1
            for i in dim:
                N *= xd0.shape[i]
        grad0 = 2 * gd0 * xp.divide(xd0 - me, N - unbiased, dtype=gd0.dtype)
        return grad0
def var(input, dim=None, unbiased=True,keepdim=False):
    return Var.apply(input, dim=dim, unbiased=unbiased, keepdim=keepdim)


class Cat(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        dim = ctx.params['dim']
        xp = cp if inputs[0].__class__ is cparray else np
        xdn=[]
        requires_grad=False
        indices=[]
        for xt in inputs:
            xdn.append(xt.data)
            if requires_grad is False and xt.requires_grad:
                requires_grad=True
            indices.append(xt.shape[dim])
        indices = np.cumsum(indices)
        yt0 = tt.tensor(xp.concatenate(xdn, dim), requires_grad=requires_grad, copy=False,_output_idx=0, grad_fn = ctx)
        ctx.params['indices'] = indices
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xp = cp if gd0.__class__ is cparray else np
        dim=ctx.params['dim']
        indices = ctx.params['indices']
        gradn=xp.split(gd0, indices_or_sections=indices[:-1], axis=dim)
        return gradn

def cat(tensors, dim=0):
    return Cat.apply(*tensors, dim=dim)

class Slice(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xd0[params['key']], requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx, _version=xd0._version)
        ctx.params['shape']=xd0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        key = ctx.params['key']
        xp = cp if gd0.__class__ is cparray else np
        grad0=xp.zeros(xd0_shape, dtype=gd0.dtype)
        grad0[key]=gd0
        return grad0


class View(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xd0.reshape(params['shape']), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx, _version=xd0._version)
        ctx.params['shape'] = xd0.shape
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0=gd0.reshape(ctx.params['shape'])
        return grad0

class Split(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        requires_grad = xt0.requires_grad
        dim=params['dim']
        split_size_or_sections=params['split_size_or_sections']
        dim_size=xd0.shape[dim]
        if dim<0:
            dim+=xd0.ndim
        if split_size_or_sections.__class__ is int:
            split_size=split_size_or_sections
            ytn = tuple(
                tt.tensor(
                    xd0[
                        tuple(
                            slice(None) if i != dim else slice(j, j + split_size) for i in range(xd0.ndim)
                        )
                    ],
                    requires_grad=requires_grad,
                    copy=False,
                    _output_idx=j//split_size,
                    grad_fn=ctx,
                    _version=xd0._version
                )
                for j in range(0, dim_size, split_size)
            )
        else:
            sections=split_size_or_sections
            if buildin_sum(sections) != dim_size:
                raise RuntimeError(f"split_with_sizes expects split_sizes to sum exactly to 8 "
                                   f"(input tensor's size at dimension {dim}), but got split_sizes={sections}")
            sum_sections = np.cumsum(split_size_or_sections)
            ytn = tuple(
                tt.tensor(
                    xd0[
                        tuple(
                            slice(None) if i != dim else slice(sum_sections[j] - sec, sum_sections[j]) for i in range(xd0.ndim)
                        )
                    ],
                    requires_grad=requires_grad,
                    copy=False,
                    _output_idx=j,
                    grad_fn=ctx,
                    _version=xd0._version
                )
                for j, sec in enumerate(sections)
            )
        ctx.save_for_backward(xt0)
        if ytn.__class__ is not tuple:
            ytn = (ytn,)
        ctx.params['output_shapes'] = tuple(yt.shape for yt in ytn)
        return ytn
    @staticmethod
    def backward(ctx, *grad_outputs):
        xt0, = ctx.saved_tensors
        xp = cp if xt0.__class__ is cparray else np
        output_shapes = ctx.params['output_shapes']
        dim = ctx.params['dim']
        grad0=xp.concatenate(
            [
                xp.zeros(output_shapes[i], dtype=xt0.dtype) if gdn is None else gdn
                for i, gdn in enumerate(grad_outputs)
            ],
            axis=dim
        )
        return grad0
def split(tensor, split_size_or_sections, dim=0):
    return Split.apply(tensor, split_size_or_sections=split_size_or_sections, dim=dim)

def chunk(input, chunks, dim=0):
    dim_size = input.shape[dim]
    if dim<0:
        dim+=input.ndim
    chunk_size = dim_size // chunks + (dim_size % chunks != 0)
    return Split.apply(input, split_size_or_sections=chunk_size, dim=dim)

class ToCopy(Function): # keep input _version: False
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        target_device = params['target_device']
        requires_grad = xt0.requires_grad
        if xd0.__class__ is cparray:
            if target_device == 'cuda':
                return xt0
            else:
                yt0=tt.tensor(xd0.get(), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        else:
            if target_device == 'cpu':
                return xt0
            else:
                if not cupy_is_loaded:
                    raise RuntimeError("cupy not installed, can't use cuda")
                yt0=tt.tensor(cparray(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        target_device = ctx.params['target_device']
        if target_device == 'cpu':
            grad0=cparray(gd0)
        else:
            grad0=tt.nparray(gd0.get())
        return grad0


class Repeat(Function): # keep input _version: False
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        sizes = params['sizes']
        requires_grad = xt0.requires_grad
        xp = cp if xd0.__class__ is cparray else np
        yt0=tt.tensor(xp.tile(xd0, sizes), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.params['shape']=xd0.shape
        ctx.params['yt0_strides']=yt0.strides
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0,=grad_outputs
        sizes = ctx.params['sizes']
        xd0_shape = ctx.params['shape']
        yt0_strides=ctx.params['yt0_strides']
        xd0_ndim=len(xd0_shape)
        leading_dims=tuple(range(len(sizes)))
        xp = cp if gd0.__class__ is cparray else np
        target_shape=sizes+xd0_shape
        target_strides = yt0_strides[:-xd0_ndim]+\
                         tuple(xd0_shape[i]*yt0_strides[i-xd0_ndim] for i in range(xd0_ndim))+ \
                         yt0_strides[-xd0_ndim:]
        grad0=xp.lib.stride_tricks.as_strided(gd0, shape=target_shape, strides=target_strides).sum(leading_dims)
        grad0 = cparray(grad0) if gd0.__class__ is cparray else nparray(grad0)
        return grad0


class Expand(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        sizes = params['sizes']
        requires_grad = xt0.requires_grad
        xp = cp if xd0.__class__ is cparray else np
        leading_dims = len(sizes) - len(xd0.shape)
        strides = [0] * leading_dims + list(xd0.strides)
        xd0_singleton_dims = [] # singleton axes to be summed during backward
        for i in range(len(sizes)):
            if i < leading_dims: # leading dimensions
                if sizes[i]<=0:
                    raise RuntimeError(f"The expanded size of the tensor ({sizes[i]}) isn't allowed in a leading, "
                                       f"non-existing dimension {i}")
            else:
                i-=len(sizes) # for non-leading dimensions, count backward
                if xd0.shape[i]==1:
                    if sizes[i]>1:
                        xd0_singleton_dims.append(i)
                        strides[i]=0
                else:
                    if sizes[i]!=-1 and xd0.shape[i]!=sizes[i]:
                        raise RuntimeError(f"The expanded size of the tensor ({sizes[i]}) must match the existing size "
                                           f"({xd0.shape[i]}) at non-singleton dimension {i+len(sizes)}.  "
                                           f"Target sizes: {sizes}.  Tensor sizes: {xd0.shape}")
        value = xp.lib.stride_tricks.as_strided(xd0, shape=sizes, strides=strides) # a numpy/cupy array
        yt0 = tt.tensor(value, requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx) # convert to nparray/cparray
        yt0._version = xd0._version # keep version
        ctx.params = {'xd0_singleton_dims':xd0_singleton_dims, 'leading_dims':leading_dims}
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_singleton_dims = tuple(ctx.params['xd0_singleton_dims'])
        leading_dims = tuple(range(ctx.params['leading_dims']))
        grad0=gd0.sum(xd0_singleton_dims+leading_dims, keepdims=True).squeeze(leading_dims)
        return grad0

class Squeeze(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        if dim.__class__ is int:
            dim = (dim,)
        if dim is None:
            dim = tuple(range(xd0.ndim))
        requires_grad = xt0.requires_grad
        squeeze_dims=tuple(i for i in dim if xd0.shape[i] == 1)
        if len(squeeze_dims) == 0:
            yt0=tt.tensor(xd0, requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx, _version=xd0._version)
        else:
            xp = cp if xd0.__class__ is cparray else np
            yt0 = tt.tensor(xp.squeeze(xd0, squeeze_dims), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx, _version=xd0._version)
        ctx.params={'squeeze_dims':squeeze_dims}
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        squeeze_dims = ctx.params['squeeze_dims']
        xp = cp if gd0.__class__ is cparray else np
        grad0 = xp.expand_dims(gd0, squeeze_dims)
        return grad0
def squeeze(input, dim=None):
    return Squeeze.apply(input, dim=dim)

class Unsqueeze(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        requires_grad = xt0.requires_grad
        xp = cp if xd0.__class__ is cparray else np
        yt0 = tt.tensor(xp.expand_dims(xd0, dim), requires_grad=requires_grad, copy=False, _output_idx=0,
                        grad_fn=ctx, _version=xd0._version)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim = ctx.params['dim']
        xp = cp if gd0.__class__ is cparray else np
        grad0 = xp.squeeze(gd0, dim)
        return grad0
def unsqueeze(input, dim):
    return Unsqueeze.apply(input, dim=dim)

class MaskedFill(Function): # keep input _version: False (except in-place)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        if xt1.ndim > 0:
            raise RuntimeError(f"masked_fill only supports a 0-dimensional value tensor, "
                               f"but got tensor with {xt1.ndim} dimension(s).")
        mask = params['mask']
        if mask.dtype.type is not np.bool_:
            raise RuntimeError(f"dtype of mask must be bool. "
                               f"Pass dtype=bool when constructing mask")

        requires_grad = xt0.requires_grad | xt1.requires_grad

        flag=False
        if xd0.__class__ is cparray and xd1.__class__ is not cparray: # xd1 is a scaler, no need to convert it to cparray
            flag=True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            raise RuntimeError(f"masked_fill: Expected inputs to be on same device")

        key=(slice(None),)*(xd0.ndim-mask.ndim)+(mask.data,)
        if params['inplace']:
            inplace_precheck(xt0)
            xd0[key]=xd1
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            xd0=xd0.copy()
            xd0[key]=xt1.data
            yt0=tt.tensor(xd0, requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.params['flag']=flag
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        mask = ctx.params['mask']
        flag = ctx.params['flag']
        leading = (slice(None),) * (gd0.ndim - mask.ndim)
        grad0, grad1 = None, None
        if ctx.needs_input_grad[1]: # grad for value. Do this first because gd0 will be changed inplace next
            key = leading + (mask.data,)
            grad1=gd0[key].sum()
            if flag:
                grad1 = nparray(grad1.get())
        if ctx.needs_input_grad[0]: # grad for input
            key = leading + (mask.data,)
            grad0 = gd0
            grad0[key] = 0
        return grad0, grad1

def masked_fill(input, mask, value):
    if value.__class__ is not tt.Tensor:
        value=tt.tensor(value, copy=False)
    return MaskedFill.apply(input, value, mask=mask, inplace=False)
def masked_fill_(input, mask, value):
    if value.__class__ is not tt.Tensor:
        value=tt.tensor(value, copy=False)
    return MaskedFill.apply(input, value, mask=mask, inplace=True)

class CopySlices(Function): # keep input _version: True (it's inplace)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        key = params['key']
        requires_grad = xt0.requires_grad | xt1.requires_grad

        # convert xd1 to same array type as xd0
        flag = None
        if xd0.__class__ is cparray and xd1.__class__ is not cparray:
            xd1 = cp.array(xd1)
            flag = True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            xd1 = xd1.get()
            flag = False

        inplace_precheck(xt0)
        xd0[key] = xd1
        yt0 = xt0
        inplace_update(yt0, requires_grad, ctx)
        ctx.params['shapes']=(xd0.shape, xd1.shape)
        ctx.params['flag']=flag
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shapes']
        key = ctx.params['key']
        flag = ctx.params['flag']
        grad0, grad1 = None, None

        if ctx.needs_input_grad[1]: # grad for value. Do this first because gd0 will be changed inplace next
            grad1 = reverse_broadcast(gd0[key], xd1_shape)
            if flag is True:
                grad1 = nparray(grad1.get())
            elif flag is False:
                grad1 = cparray(grad1)
        if ctx.needs_input_grad[0]:  # grad for input
            grad0 = gd0
            grad0[key] = 0
        return grad0, grad1

class Copy(Function): # keep input _version: True (it's inplace)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        requires_grad = xt0.requires_grad | xt1.requires_grad

        # convert xd1 to same array type as xd0
        flag = None
        if xd0.__class__ is cparray and xd1.__class__ is not cparray:
            xd1 = cp.array(xd1)
            flag = True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            xd1 = xd1.get()
            flag = False

        inplace_precheck(xt0)
        xd0[...] = xd1
        yt0 = xt0
        inplace_update(yt0, requires_grad, ctx)
        ctx.params['flag'] = flag
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0, grad1 = None, None
        flag = ctx.params['flag']
        if ctx.needs_input_grad[1]: # grad for value.
            grad1 = gd0
            if flag is True:
                grad1 = nparray(grad1.get())
            elif flag is False:
                grad1 = cparray(grad1)
        if ctx.needs_input_grad[0]: # grad for input, zero
            grad0 = gd0
            grad0[...] = 0
        return grad0, grad1 # no grad for input
