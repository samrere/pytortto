from tortto import np, cp, cparray, cupy_is_loaded, _int_zero
from .function import *
from .helper import *

buildin_sum = sum

class Permute(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xp.transpose(xd0, params['dims']), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.params = params
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

class Transpose(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        dim0, dim1 = params['dim0'], params['dim1']
        yt0 = tt.tensor(xp.swapaxes(xd0, dim0,dim1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        ctx.params = params
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
        ctx.params = {'shape': xd0.shape, 'dim':dim, 'keepdim':keepdim}
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
        ctx.params = params
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
        ctx.params = params
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
        params['indices']=indices
        yt0 = tt.tensor(xp.concatenate(xdn, dim), requires_grad=requires_grad, copy=False,_output_idx=0, grad_fn = ctx)
        ctx.params = params
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

class Slice(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xd0[params['key']], requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        params['shape']=xd0.shape
        ctx.params = params
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


class View(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xd0.reshape(params['shape']), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn = ctx)
        params['shape'] = xd0.shape
        ctx.params = params
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0=gd0.reshape(ctx.params['shape'])
        return grad0

class Split(Function):
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
                    grad_fn=ctx
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
                    grad_fn=ctx
                )
                for j, sec in enumerate(sections)
            )
        params['shape'] = xd0.shape
        ctx.params = params
        if ytn.__class__ is not tuple:
            ytn = (ytn,)
        return ytn
    @staticmethod
    def backward(ctx, *grad_outputs):
        ...
def split(tensor, split_size_or_sections, dim=0):
    return Split.apply(tensor, split_size_or_sections=split_size_or_sections, dim=dim)





def chunk(x, chunks, dim=0):
    x_data = x.data
    dim_size = x_data.shape[dim]
    if dim < 0:
        dim += x_data.ndim
    chunk_size = dim_size // chunks + (dim_size % chunks != 0)
    return tuple(x[tuple(slice(None) if i != dim else slice(j, j + chunk_size) for i in range(x_data.ndim))] for j in
                 range(0, dim_size, chunk_size))

def _cuda(x):
    if x.data.__class__ is cparray:
        return x
    else:
        if not cupy_is_loaded:
            raise RuntimeError("cupy not installed, can't use cuda")
        value = cparray(x.data)
        return build_links(value, x.requires_grad, _cuda, x)


@register_gradients(_cuda)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad.get()


def _cpu(x):
    if x.data.__class__ is cparray:
        value = x.data.get()
        return build_links(value, x.requires_grad, _cpu, x)
    else:
        return x


@register_gradients(_cpu)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += cp.array(grad)


def _repeat(x, *sizes):
    x_data = x.data
    xp = cp if x_data.__class__ is cparray else np
    value = xp.tile(x_data, sizes)
    return build_links(value, x.requires_grad, _repeat, x, sizes=sizes, x_shape=x_data.shape)


@register_gradients(_repeat)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        sizes = params['sizes']
        x_shape = params['x_shape']
        new_shape = []
        sum_axes = []
        idx = 0
        for i in range(-len(sizes), 0):
            if -i > len(x_shape):
                new_shape.append(sizes[i])
                sum_axes.append(idx)
                idx += 1
            else:
                if sizes[i] == 1:
                    new_shape.append(x_shape[i])
                    idx += 1
                else:
                    new_shape.extend([sizes[i], x_shape[i]])
                    sum_axes.append(idx)
                    idx += 2
        inputs[0].grad += grad.reshape(new_shape).sum(tuple(sum_axes))


def _expand(x, *dims):
    x_data = x.data
    x_shape = x_data.shape
    leading_dim = len(dims) - len(x_shape)
    dims = np.array(dims)
    x_shape = np.array((1,) * leading_dim + x_data.shape)  # add singleton to match dims
    singleton = np.logical_and(x_shape == 1, dims > 1)
    dims[~singleton] = x_shape[~singleton]  # new shape
    strides = np.array((0,) * leading_dim + x_data.strides)
    strides[singleton] = 0
    xp = cp if x_data.__class__ is cparray else np
    value = xp.lib.stride_tricks.as_strided(x_data, shape=dims, strides=strides)
    return build_links(value, x.requires_grad, _expand, x, sum_axes=tuple(np.arange(len(dims))[singleton]),
                       leading_dim=tuple(range(leading_dim)))


@register_gradients(_expand)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        sum_axes = params['sum_axes']
        leading_dim = params['leading_dim']
        inputs[0].grad += grad.sum(sum_axes, keepdims=True).squeeze(leading_dim)


def squeeze(x, dim=None):
    x_data = x.data
    x_shape = x_data.shape
    if dim.__class__ is int:
        dim = (dim,)
    if dim is None:
        dim = tuple(range(x_data.ndim))
    unchanged = False
    dim = tuple(i for i in dim if x_shape[i] == 1)
    if len(dim) == 0:
        value = x_data.copy()
        unchanged = True
    else:
        xp = cp if x_data.__class__ is cparray else np
        value = xp.squeeze(x_data, dim)
    return build_links(value, x.requires_grad, squeeze, x, dim=dim, unchanged=unchanged)


@register_gradients(squeeze)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        unchanged = params['unchanged']
        if unchanged:
            inputs[0].grad += grad
        else:
            xp = cp if grad.__class__ is cparray else np
            dim = params['dim']  # dim is a tuple
            inputs[0].grad += xp.expand_dims(grad, dim)


def unsqueeze(x, dim):
    x_data = x.data
    xp = cp if x_data.__class__ is cparray else np
    value = xp.expand_dims(x_data, dim)
    return build_links(value, x.requires_grad, unsqueeze, x, dim=dim)


@register_gradients(unsqueeze)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        dim = params['dim']
        xp = cp if grad.__class__ is cparray else np
        inputs[0].grad += xp.squeeze(grad, dim)


def masked_fill(x, mask, val):
    value = x.data.copy()
    xp = cp if value.__class__ is cparray else np
    mask = xp.lib.stride_tricks.as_strided(mask.data, shape=value.shape,
                                           strides=(0,) * (value.ndim - mask.ndim) + mask.strides)
    value[mask] = val
    return build_links(value, x.requires_grad, masked_fill, x, mask=mask)


@register_gradients(masked_fill)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        mask = ~params['mask']  # take not to mask, take grad that are not masked
        xp = cp if grad.__class__ is cparray else np
        if inputs[0].grad is _int_zero:
            value = xp.zeros_like(grad)
            value[mask] = grad[mask]
            inputs[0].grad += value
        else:
            inputs[0].grad[mask] += grad[mask]
