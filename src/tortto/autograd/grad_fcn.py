from tortto import np, cp, cp_ndarray, cupy_is_loaded, _int_zero
from .helper import *

buildin_sum = sum


def transpose(x, axes=None):
    xp = cp if x.data.__class__ is cp_ndarray else np
    output = build_links(xp.transpose(x.data, axes), x.requires_grad, transpose, x, axes=axes)
    return output


@register_gradients(transpose)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        if params is None:
            inputs[0].grad += xp.transpose(grad)
        else:
            inputs[0].grad += xp.transpose(grad, axes=np.argsort(params['axes']))


def swapaxes(x, axis1, axis2):
    ndim = x.ndim
    if not -ndim<=axis1<=(ndim-1):
        raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {axis1})')
    if not -ndim<=axis2<=(ndim-1):
        raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {axis2})')
    axes = np.arange(ndim)
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return transpose(x, axes)

def moveaxis(x, source, destination):
    ndim = x.ndim
    if not -ndim<=source<=(ndim-1):
        raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {source})')
    if not -ndim<=destination<=(ndim-1):
        raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {destination})')
    if destination < 0:
        destination += ndim
    axes = list(range(ndim))
    axes.insert(destination, axes.pop(source))
    return transpose(x, axes)

def matmul(x0, x1):
    x0_data = x0.data
    x1_data = x1.data
    if x0_data.ndim==0 or x1_data.ndim==0:
        raise RuntimeError(f'both arguments to matmul need to be at least 1D, but they are {x0_data.ndim}D and {x1_data.ndim}D')
    x0_low_dim = x0_data.ndim < 2
    x1_low_dim = x1_data.ndim < 2
    x0_true_shape = (1,) + x0_data.shape if x0_low_dim else x0_data.shape
    x1_true_shape = x1_data.shape + (1,) if x1_low_dim else x1_data.shape
    if x0_true_shape[-1] != x1_true_shape[-2]:
        raise ValueError(
            f'Dimension mismatch: input0 has a shape of {x0.data.shape} and input1 has a shape of {x1.data.shape}')
    value = x0_data @ x1_data
    output = build_links(value, x0.requires_grad | x1.requires_grad, matmul, x0, x1, x0_low_dim=x0_low_dim,
                         x1_low_dim=x1_low_dim)
    return output


@register_gradients(matmul)
def backward(tensor, grad, params):
    inputs = tensor.parents
    x0 = inputs[0]
    x1 = inputs[1] if len(inputs) > 1 else inputs[0]
    x0_low_dim = params['x0_low_dim']
    x1_low_dim = params['x1_low_dim']
    if x1_low_dim:
        grad = grad[..., None]
        x1_data = x1.data[:, None]
    else:
        x1_data = x1.data
    if x0_low_dim:
        grad = grad[..., None, :]
        x0_data = x0.data[None]
    else:
        x0_data = x0.data

    if x0.requires_grad:
        result = grad @ x1_data.swapaxes(-1, -2)
        x0.grad += reverse_broadcast(result, x0.shape)
    if x1.requires_grad:
        result = x0_data.swapaxes(-1, -2) @ grad
        if x1_low_dim:
            axis0 = tuple(range(result.ndim - x1_data.ndim))
            x1.grad += result.sum(axis=axis0)[..., 0]
        else:
            x1.grad += reverse_broadcast(result, x1.shape)


def sum(x, dim=None, keepdims=False):
    # if axis is None and keepdims is False:
    #     raise RuntimeError('taking sum of all axes, but keepdims is False.')
    x_data=x.data
    xp = cp if x_data.__class__ is cp_ndarray else np
    if dim is None:
        dim=tuple(range(x_data.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    value = xp.sum(x_data, axis=dim, keepdims=keepdims)
    output = build_links(value, x.requires_grad, sum, x, dim=dim, keepdims=keepdims)
    return output


@register_gradients(sum)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        if params is None:  # if axis is None, repeat grad in all dimensions of x.shape
            inputs[0].grad += xp.lib.stride_tricks.as_strided(grad, shape=x.shape, strides=[0 for _ in x.shape])
        else:
            dim = params['dim']
            keepdims = params['keepdims']
            if not keepdims:
                grad = xp.expand_dims(grad, dim)
            strides = list(grad.strides)
            for i in dim:
                strides[i] = 0  # repeat along axis in x.shape
            inputs[0].grad += xp.lib.stride_tricks.as_strided(grad, shape=x.shape, strides=strides)


def mean(x, dim=None, keepdims=False):
    # if axis is None and keepdims is False:
    #     raise RuntimeError('taking mean of all axes, but keepdims is False.')
    x_data = x.data
    xp = cp if x_data.__class__ is cp_ndarray else np
    if dim is None:
        dim=tuple(range(x_data.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    value = xp.mean(x_data, axis=dim, keepdims=keepdims)
    output = build_links(value, x.requires_grad, mean, x, dim=dim, keepdims=keepdims)
    return output


@register_gradients(mean)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        if params is None:  # if axis is None, repeat grad in all dimensions of x.shape
            inputs[0].grad += xp.lib.stride_tricks.as_strided(xp.divide(grad, x.size, dtype=grad.dtype), shape=x.shape, strides=[0 for _ in x.shape])
        else:
            dim = params['dim']
            keepdims = params['keepdims']
            if not keepdims:
                grad = xp.expand_dims(grad, dim)
            N = 1
            strides = list(grad.strides)
            for i in dim:
                N *= x.shape[i]
                strides[i] = 0  # repeat along axis in x.shape
            inputs[0].grad += xp.lib.stride_tricks.as_strided(xp.divide(grad, N, dtype=grad.dtype), shape=x.shape, strides=strides)


def var(x, dim=None, unbiased=True, keepdims=False):
    # if axis is None and keepdims is False:
    #     raise RuntimeError('taking var of all axes, but keepdims is False.')
    x_data = x.data
    xp = cp if x_data.__class__ is cp_ndarray else np
    if dim is None:
        dim=tuple(range(x_data.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    ddof = unbiased == True
    value = xp.var(x_data, axis=dim, ddof=ddof, keepdims=keepdims)
    output = build_links(value, x.requires_grad, var, x, ddof=ddof, dim=dim, keepdims=keepdims)
    return output


@register_gradients(var)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        dim = params['dim']
        ddof = params['ddof']
        keepdims = params['keepdims']
        if not keepdims:
            grad = xp.expand_dims(grad, dim)
        x = inputs[0].data
        me = xp.mean(x, axis=dim, keepdims=True)
        if params is None:  # if axis is None
            N = x.size  # all element
        else:
            N = 1
            for i in dim:
                N *= x.shape[i]
        inputs[0].grad += 2 * grad * xp.divide(x - me, N - ddof, dtype=grad.dtype)


def cat(tensors, dim=0):
    xp = cp if tensors[0].data.__class__ is cp_ndarray else np
    requires_grad = any(t.requires_grad for t in tensors)
    tensors_data = tuple(t.data for t in tensors)
    indices = np.cumsum([t.shape[dim] for t in tensors_data])
    value = xp.concatenate(tensors_data, dim)
    output = build_links(value, requires_grad, cat, *tensors, axis=dim, indices=indices)
    return output


@register_gradients(cat)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    axis = params['axis']
    indices = params['indices']
    grad = xp.split(grad, indices_or_sections=indices, axis=axis)
    for idx, inpt in enumerate(inputs):
        if inpt.requires_grad:
            inpt.grad += grad[idx]


def _slice(x, key):
    return build_links(x.data[key], x.requires_grad, _slice, x, key=key)


@register_gradients(_slice)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        if inputs[0].grad is _int_zero:
            inputs[0].grad = xp.zeros_like(inputs[0].data)
        inputs[0].grad[params['key']] += grad


def _view(x, newshape):
    return build_links(x.data.reshape(newshape), x.requires_grad, _view, x, newshape=newshape)


@register_gradients(_view)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad.reshape(inputs[0].shape)


def chunk(x, chunks, dim=0):
    x_data = x.data
    dim_size = x_data.shape[dim]
    if dim < 0:
        dim += x_data.ndim
    chunk_size = dim_size // chunks + (dim_size % chunks != 0)
    return tuple(x[tuple(slice(None) if i != dim else slice(j, j + chunk_size) for i in range(x_data.ndim))] for j in
                 range(0, dim_size, chunk_size))


def split(x, split_size_or_sections, dim=0):
    x_data = x.data
    dim_size = x_data.shape[dim]
    if dim < 0:
        dim += x_data.ndim
    if split_size_or_sections.__class__ is int:
        return tuple(
            x[tuple(slice(None) if i != dim else slice(j, j + split_size_or_sections) for i in range(x_data.ndim))] for
            j in range(0, dim_size, split_size_or_sections))
    else:
        if buildin_sum(split_size_or_sections) != dim_size:
            raise RuntimeError(
                f'sum of split sections {split_size_or_sections} does not equal to dimension size {dim_size}.')
        sum_sections = np.cumsum(split_size_or_sections)
        return tuple(x[tuple(
            slice(None) if i != dim else slice(sum_sections[j] - sec, sum_sections[j]) for i in range(x_data.ndim))] for
                     j, sec in enumerate(split_size_or_sections))


def _cuda(x):
    if x.data.__class__ is cp_ndarray:
        return x
    else:
        if not cupy_is_loaded:
            raise RuntimeError("cupy not installed, can't use cuda")
        value = cp.array(x.data)
        return build_links(value, x.requires_grad, _cuda, x)


@register_gradients(_cuda)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad.get()


def _cpu(x):
    if x.data.__class__ is cp_ndarray:
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
    xp = cp if x_data.__class__ is cp_ndarray else np
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
    xp = cp if x_data.__class__ is cp_ndarray else np
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
        xp = cp if x_data.__class__ is cp_ndarray else np
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
            xp = cp if grad.__class__ is cp_ndarray else np
            dim = params['dim']  # dim is a tuple
            inputs[0].grad += xp.expand_dims(grad, dim)


def unsqueeze(x, dim):
    x_data = x.data
    xp = cp if x_data.__class__ is cp_ndarray else np
    value = xp.expand_dims(x_data, dim)
    return build_links(value, x.requires_grad, unsqueeze, x, dim=dim)


@register_gradients(unsqueeze)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        dim = params['dim']
        xp = cp if grad.__class__ is cp_ndarray else np
        inputs[0].grad += xp.squeeze(grad, dim)


def masked_fill(x, mask, val):
    value = x.data.copy()
    xp = cp if value.__class__ is cp_ndarray else np
    mask = xp.lib.stride_tricks.as_strided(mask.data, shape=value.shape,
                                           strides=(0,) * (value.ndim - mask.ndim) + mask.strides)
    value[mask] = val
    return build_links(value, x.requires_grad, masked_fill, x, mask=mask)


@register_gradients(masked_fill)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        mask = ~params['mask']  # take not to mask, take grad that are not masked
        xp = cp if grad.__class__ is cp_ndarray else np
        if inputs[0].grad is _int_zero:
            value = xp.zeros_like(grad)
            value[mask] = grad[mask]
            inputs[0].grad += value
        else:
            inputs[0].grad[mask] += grad[mask]
