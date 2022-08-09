""""""
"""
datatype inconsistencies in numpy:
1. a small float32 / large int will result in float64:
    x=np.array([0.01],dtype=np.float32)
    y=x/(128*32*32)
    y.dtype --> float64
2. operations on float32 array scalar will result in float64
    x=np.float32(3)
    y=x+2 # or x*2 or x/2 ...
    y.dtype --> float64
solution:
use ufuncs such as np.add, np.divide, ... and specify dtype:
    x=np.array([0.01],dtype=np.float32)
    y=np.divide(x, 128*32*32, dtype=np.float32)

DO NOT: don't use y=x/np.array(128*32*32, dtype=np.float32). it's slow in cupy
########################################################################################################################
contiguous issue: conv2d and conv_transpose2d is slow when incoming data is not contiguous:
```
import tortto as tt
x=tt.randn(64,112,112,64).swapaxes(1,-1)
```
```
%%time
m=tt.nn.Conv2d(64,128,3)
y=m(x) -> Wall time: 5.59 s 
```
```
%%time
m=tt.nn.Conv2d(64,128,3)
x=x.contiguous()
y=m(x) -> Wall time: 1.84 s
```
solution: use tensor.contiguous() on input data.
"""
import math
from tortto import *

scipy_is_loaded = bool(find_spec('scipy'))
sparse_is_loaded = cupy_is_loaded or scipy_is_loaded
if cupy_is_loaded:
    import cupyx as cpx
    import cupy.fft as cp_fft
    import cupyx.scipy.special as cp_special
    import cupyx.scipy.sparse as cp_sparse

#
HALF = {float16: float16(.5), float32: float32(.5), float64: float64(.5)}
ONE = {float16: float16(1), float32: float32(1), float64: float64(1)}
TWO = {float16: float16(2), float32: float32(2), float64: float64(2)}
SQRT_2 = {float16: float16(math.sqrt(2)), float32: float32(math.sqrt(2)), float64: float64(math.sqrt(2))}
SQRT_2_over_PI = {float16: float16(math.sqrt(2 / math.pi)), float32: float32(math.sqrt(2 / math.pi)),
                  float64: float64(math.sqrt(2 / math.pi))}
CONST = {float16: float16(.08943), float32: float32(.08943), float64: float64(.08943)}
_floating_point = {float16, float32, float64}

# default functions using numpy
np_erf = lambda x: np.tanh(SQRT_2_over_PI[x.dtype.type] * (x + CONST[x.dtype.type] * x * x * x))
np_rfft2 = np.fft.rfft2
np_irfft2 = np.fft.irfft2
# if scipy is present, replace default numpy functions to scipy
if scipy_is_loaded:
    from scipy.fft import rfft2 as np_rfft2, irfft2 as np_irfft2
    from scipy.special import erf as np_erf
    import scipy.sparse as sci_sparse

# math has prod since python 3.8, math.prod is 30 times faster
prod = math.prod if hasattr(math, 'prod') else np.prod


def relu(inpt: Tensor, inplace=False):
    x = inpt.data
    if inplace:
        x *= x > 0
        value = x
    else:
        value = x * (x > 0)
    output = build_links(value, inpt.requires_grad, relu, inpt)
    return output


@register_gradients(relu)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad * (inputs[0].data > 0)


def leaky_relu(inpt: Tensor, negative_slope=0.01, inplace=False):
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    value = xp.maximum(x * negative_slope, x, out=x if inplace else None)
    output = build_links(value, inpt.requires_grad, leaky_relu, inpt, negative_slope=negative_slope)
    return output


@register_gradients(leaky_relu)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    negative_slope = params['negative_slope']
    if inputs[0].requires_grad:
        value = xp.ones_like(grad)
        value[inputs[0].data < 0] = negative_slope
        inputs[0].grad += grad * value


def gelu(inpt):
    x = inpt.data
    if x.dtype.type not in _floating_point:
        raise RuntimeError(f'gelu not implemented for {x.dtype.type.__name__}')
    s = x / SQRT_2[x.dtype.type]
    # select erf
    erf = cp_special.erf if x.__class__ is cp_ndarray else np_erf
    erf_s = erf(s)
    value = HALF[x.dtype.type] * x * (erf_s + ONE[x.dtype.type])
    output = build_links(value, inpt.requires_grad, gelu, inpt, s=s, erf_s=erf_s)
    return output


@register_gradients(gelu)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    x = inputs[0].data
    s = params['s']
    erf_s = params['erf_s']
    if inputs[0].requires_grad:
        pdf_s = SQRT_2_over_PI[x.dtype.type] * xp.exp(-(s * s))
        half = HALF[x.dtype.type]
        inputs[0].grad += grad * (half + half * erf_s + ((half * x * pdf_s) / SQRT_2[x.dtype.type]))


def mse_loss(inpt: Tensor, target: Tensor, reduction='mean'):
    if inpt.shape != target.shape:
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.shape, inpt.shape), stacklevel=2, )
    loss = (inpt.data - target.data) ** 2
    if reduction == 'mean':
        loss = loss.mean()  # note: loss is now a numpy array not Tensor
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    output = build_links(loss, inpt.requires_grad, mse_loss, inpt, reduction=reduction, target=target)
    return output


@register_gradients(mse_loss)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    target = params['target']
    if inputs[0].requires_grad:
        reduction = params['reduction']
        value = 2 * grad * (inputs[0].data - target.data)
        if reduction == 'none':
            inputs[0].grad += value
        elif reduction == 'sum':
            inputs[0].grad += value
        elif reduction == 'mean':
            inputs[0].grad += xp.divide(value, inputs[0].numel(), dtype=grad.dtype)


def binary_cross_entropy(inpt, target, weight=None, reduction='mean'):
    if target.shape != inpt.shape:
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.shape, inpt.shape))
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    y = target.data
    w = weight.data if weight is not None else 1
    loss = -w * (y * xp.clip(xp.log(x), -100, None) + (1 - y) * xp.clip(xp.log(1-x), -100, None))
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    output = build_links(loss, inpt.requires_grad, binary_cross_entropy, inpt, reduction=reduction, weight=w,
                         target=target)
    return output


@register_gradients(binary_cross_entropy)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    x = inputs[0].data
    y = params['target'].data
    w = params['weight']
    if inputs[0].requires_grad:
        reduction = params['reduction']
        # value = grad * -w * (y / x - (1 - y) / (1 - x))
        value = grad * w * (x-y) * xp.clip(1 / x, None, 1e12) * xp.clip(1 / (1-x), None, 1e12)
        if reduction == 'none':
            inputs[0].grad += value
        elif reduction == 'sum':
            inputs[0].grad += value
        elif reduction == 'mean':
            inputs[0].grad += xp.divide(value, inputs[0].numel(), dtype=grad.dtype)


def binary_cross_entropy_with_logits(inpt, target, weight=None, pos_weight=None, reduction='mean'):
    if target.shape != inpt.shape:
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.shape, inpt.shape))
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    y = target.data
    w = weight.data if weight is not None else 1
    p = pos_weight.data if pos_weight is not None else 1
    log_sigmoid = x * (x < 0) - xp.log1p(xp.exp(-xp.abs(x)))
    loss = w * ((y * (1 - p) - 1) * log_sigmoid + x - x * y)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    output = build_links(loss, inpt.requires_grad, binary_cross_entropy_with_logits, inpt, reduction=reduction,
                         weight=w, target=target, pos_weight=p, log_sigmoid=log_sigmoid)
    return output


@register_gradients(binary_cross_entropy_with_logits)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    x = inputs[0].data
    y = params['target'].data
    log_sigmoid = params['log_sigmoid']
    w = params['weight']
    p = params['pos_weight']
    if inputs[0].requires_grad:
        reduction = params['reduction']
        value = grad * (w * (y * (1 - p) - 1) * xp.exp(-x + log_sigmoid) + w * (1 - y))
        if reduction == 'none':
            inputs[0].grad += value
        elif reduction == 'sum':
            inputs[0].grad += value
        elif reduction == 'mean':
            inputs[0].grad += xp.divide(value, inputs[0].numel(), dtype=grad.dtype)


def nll_loss(inpt, target, weight=None, ignore_index=-100, reduction='mean'):
    if not np.issubdtype(target.dtype, np.integer):
        raise RuntimeError(
            f'expected scalar type Int but found {target.dtype}. Use "dtype=int" when creating target tensor.')
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    y = target.data
    if weight is None:
        w = xp.ones((1, x.shape[1]), dtype=bool)
    else:
        w = weight.data
    dim = x.ndim
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(x.shape[0], y.shape[0]))
    if dim == 2:  # expand x dim to at least 3
        x = x[..., None]
    if y.ndim == 1:  # expand y dim to at least 2
        y = y[..., None]
    if y.shape[1:] != x.shape[2:]:
        raise ValueError("Expected target size {}, got {}".format(x.shape[2:], y.shape))

    ignored = (y != ignore_index)
    idx = np.indices(y.shape, sparse=True)
    criteria = (idx[0], y, *idx[1:])
    coef = w[0, y] * ignored
    loss = -x[criteria] * coef
    N = None
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        N = xp.count_nonzero(ignored)
        loss = xp.divide(loss.sum(), N, dtype=x.dtype)
    elif reduction == 'none':
        pass
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    output = build_links(loss, inpt.requires_grad, nll_loss, inpt, reduction=reduction, coef=coef, criteria=criteria,
                         N=N)
    return output


@register_gradients(nll_loss)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    x = inputs[0].data
    if x.ndim == 2:
        x = x[..., None]
    criteria = params['criteria']
    coef = params['coef']
    if inputs[0].requires_grad:
        reduction = params['reduction']
        value = -grad * coef
        if reduction == 'mean':
            N = params['N']
            value = xp.divide(value, N, dtype=grad.dtype)
        inputs[0].grad = xp.zeros_like(x)
        inputs[0].grad[criteria] = value
        if inputs[0].data.ndim == 2:  # if ndim of original input is 2, delete the last axis
            inputs[0].grad = inputs[0].grad[..., 0]


def softmax(inpt: Tensor, dim=None):
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    axis_aug = x - xp.max(x, axis=dim, keepdims=True)
    axis_exp = xp.exp(axis_aug)
    axis_sum_exp = xp.sum(axis_exp, axis=dim, keepdims=True)
    output = build_links(axis_exp / axis_sum_exp, inpt.requires_grad, softmax, inpt, axis=dim)
    return output


@register_gradients(softmax)
def backward(tensor: Tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        dim = params['axis']
        y = tensor.data
        inputs[0].grad += (grad - (grad * y).sum(dim, keepdims=True)) * y


def log_softmax(inpt: Tensor, dim=None):
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    axis_aug = x - xp.max(x, axis=dim, keepdims=True)
    axis_exp = xp.exp(axis_aug)
    axis_sum_exp = xp.sum(axis_exp, axis=dim, keepdims=True)
    axis_log_sum_exp = xp.log(axis_sum_exp)
    output = build_links(axis_aug - axis_log_sum_exp, inpt.requires_grad, log_softmax, inpt, axis=dim,
                         softmax=axis_exp / axis_sum_exp)
    return output


@register_gradients(log_softmax)
def backward(tensor: Tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        dim = params['axis']
        y = params['softmax']
        inputs[0].grad += grad - grad.sum(axis=dim, keepdims=True) * y


def logsigmoid(inpt):
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    value = x * (x < 0) - xp.log1p(xp.exp(-xp.abs(x)))
    output = build_links(value, inpt.requires_grad, logsigmoid, inpt)
    return output


@register_gradients(logsigmoid)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        inputs[0].grad += grad * xp.exp(-x + tensor.data)


def linear(inpt: Tensor, weight: Tensor, bias=None):
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    # linear is defined as y=X@A.T+b
    x = inpt.data
    value = x @ weight.data.T
    requires_grad = inpt.requires_grad | weight.requires_grad
    if bias is not None:
        value += bias.data
        requires_grad |= bias.requires_grad
    output = build_links(value, requires_grad, linear, inpt, weight, bias)
    return output


@register_gradients(linear)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    x = inputs[0]  # input
    weight = inputs[1]  # weight
    bias = inputs[2]  # bias, can be None
    if x.requires_grad:
        x.grad += grad @ weight.data
    if weight.requires_grad:  # weight
        # TODO: follow up https://github.com/cupy/cupy/issues/6673
        # not yet working for cupy: weight.grad += xp.einsum('...j,...k->jk', grad, x.data, optimize=True)
        weight.grad += xp.tensordot(grad, x.data, axes=(np.arange(grad.ndim - 1), np.arange(grad.ndim - 1)))
    if bias and bias.requires_grad:  # bias
        bias.grad += grad.sum(tuple(range(grad.ndim - 1)))


def pad(inpt, padding, mode='constant', value=0.0):
    '''
    TODO: implement cache
    TODO: only support padding with constant for now
    '''
    assert len(padding) % 2 == 0, "Padding length must be divisible by 2"
    assert len(padding) // 2 <= inpt.dim(), "Padding length too large"
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    padding_dims = len(padding) // 2  # number of dimensions that needs padding
    pad_width = tuple((0, 0) if i >= padding_dims else padding[(2 * i):(2 * i + 2)] for i in reversed(range(x.ndim)))
    if mode == 'constant':
        value = xp.pad(x, pad_width, mode='constant', constant_values=value)
    else:
        raise NotImplementedError(
            'TODO: Only support padding with constant for now, need to implement reflect, replicate and circular mode')
    return build_links(value, inpt.requires_grad, pad, inpt, mode=mode, pad_width=pad_width)


@register_gradients(pad)
def backward(tensor: Tensor, grad, params):
    inputs = tensor.parents
    inpt = inputs[0]
    if inpt.requires_grad:
        x_shape = inpt.shape
        pad_width = params['pad_width']
        selection = tuple(slice(pad_width[i][0], pad_width[i][0] + x_shape[i]) for i in range(inpt.ndim))
        inpt.grad += grad[selection]


def _strided_split(xp, arr, split_axis, groups):
    group_size = arr.shape[split_axis] // groups
    return xp.lib.stride_tricks.as_strided(arr, shape=(
        *arr.shape[:split_axis], groups, group_size, *arr.shape[(split_axis + 1):]), strides=(
        *arr.strides[:split_axis], group_size * arr.strides[split_axis], arr.strides[split_axis],
        *arr.strides[(split_axis + 1):])), group_size


def _strided_repeat(xp, arr, repeat_axis, repeat):
    return xp.lib.stride_tricks.as_strided(arr, shape=(*arr.shape[:repeat_axis], repeat, *arr.shape[repeat_axis:]),
                                           strides=(*arr.strides[:repeat_axis], 0, *arr.strides[repeat_axis:]))


def _calc_pad_shape(inpt, weight_dilated, padding):
    full_conv_shapes_h = inpt.shape[-2] + weight_dilated.shape[-2] - 1 + padding[0] * 2
    full_conv_shapes_w = inpt.shape[-1] + weight_dilated.shape[-1] - 1 + padding[1] * 2
    return (full_conv_shapes_h, full_conv_shapes_w)


_PADDED = dict()
_DILATED = dict()


def cache(target):
    def decorator(fn):
        def wrapper(xp, a, *args):
            # early exit if no padding or no dilation
            ######################################
            ha, wa = args[0]
            if target is _PADDED:
                he, we = args[1] if len(args) > 1 else (0, 0)
                if ha + wa + he + we == 0:
                    return a
            if target is _DILATED and ha + wa == 2:
                return a
            #######################################
            key = (xp, *a.shape, *args)
            out = target.get(key)
            no_cache = args[-1]
            if out is None:  # not in cache
                out = fn(xp, a, *args)
                if not no_cache:  # save new array into cache
                    target[key] = out
            else:  # if in cache, replace with new values
                assert no_cache == False, 'no_cache is False but array exists in cache! bug!!'
                h, w = a.shape[-2:]
                if target is _PADDED:
                    out[..., ha:(ha + h), wa:(wa + w)] = a
                elif target is _DILATED:
                    out[..., ::ha, ::wa] = a
                else:
                    raise NotImplementedError('update rule is not implemented')
            return out

        return wrapper

    return decorator


@cache(target=_DILATED)
def _dilate(xp, arr, dilation, no_cache=False):
    '''
    no_cache: don't save this array into cache.
    It's important for padded or dilated arrays that passed to `build_links` to be left untouched.
    '''
    hd, wd = dilation
    hk, wk = arr.shape[-2:]
    dilated = xp.zeros((*arr.shape[:-2], (hk - 1) * hd + 1, (wk - 1) * wd + 1), dtype=arr.dtype)
    dilated[..., ::hd, ::wd] = arr
    return dilated


@cache(target=_PADDED)
def _padding(xp, arr, padding, extra_padding=(0, 0), aug=False, val=0, no_cach=False):
    '''
    internal function only used in Conv2d and ConvTransposed2d, different from pad()
    no_cache: don't save this array into cache.
    It's important for padded or dilated arrays that passed to `build_links` to be left untouched.
    '''
    hp, wp = padding
    he, we = extra_padding
    pad_width = [[0, 0] for _ in range(arr.ndim)]
    if aug:
        pad_width[-2] = [hp, 0]
        pad_width[-1] = [wp, 0]
    else:
        pad_width[-2] = [hp, hp + he]
        pad_width[-1] = [wp, wp + we]
    padded = xp.pad(arr, pad_width, constant_values=val)
    return padded


def _pad_right(xp, arr, shape, padding):
    hp, wp = padding
    ho, wo = shape
    pad_width = [[0, 0] for _ in range(arr.ndim)]
    pad_width[-2] = [hp, ho - arr.shape[-2] - hp]
    pad_width[-1] = [wp, wo - arr.shape[-1] - wp]
    padded = xp.pad(arr, pad_width)
    return padded


def _sliding_window_view_groups(xp, padded, weight, stride, dilation):
    g, C_in, H_in, W_in = padded.shape[-4:]
    H_out = math.floor((H_in - dilation[0] * (weight.shape[-2] - 1) - 1) / stride[0] + 1)
    W_out = math.floor((W_in - dilation[1] * (weight.shape[-1] - 1) - 1) / stride[1] + 1)
    shape = (*padded.shape[:-4], H_out, W_out, weight.shape[-5], padded.shape[-3], *weight.shape[-2:])  # NHWg*hw
    strides = (*padded.strides[:-4],  # batch
               padded.strides[-2] * stride[0],  # H dimension
               padded.strides[-1] * stride[1],  # W dimension
               padded.strides[1],  # groups
               padded.strides[-3],  # input channel
               padded.strides[-2] * dilation[0],  # kernel height
               padded.strides[-1] * dilation[1],  # kernel width
               )

    expanded = xp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return expanded


def _sliding_window_view(xp, x, kernel_size, stride, dilation, padding=(0, 0), ceil_mode=False, val=0):
    aug = False
    C_in, H_in, W_in = x.shape[-3:]
    hw, ww = kernel_size[-2:]
    hs, ws = stride
    hp, wp = padding
    hd, wd = dilation
    fcn = math.ceil if ceil_mode else math.floor
    H_out = fcn((H_in + 2 * hp - hd * (hw - 1) - 1) / hs + 1)
    W_out = fcn((W_in + 2 * wp - wd * (ww - 1) - 1) / ws + 1)
    if ceil_mode:  # calculate extra padding to the left and down for ceil mode
        he = (H_out - 1) * hs - (H_in + 2 * hp - hd * (hw - 1) - 1)
        we = (W_out - 1) * ws - (W_in + 2 * wp - wd * (ww - 1) - 1)
        # augmentation
        if he + hp >= (hw - 1) * hd + 1:  # dilated weight size
            aug = True
            H_out -= 1
        if we + wp >= (ww - 1) * wd + 1:
            aug = True
            W_out -= 1
    else:
        he, we = 0, 0

    # pad input
    padded = _padding(xp, x, padding, (he, we), aug, val)
    shape = (*x.shape[:-3], H_out, W_out, padded.shape[-3], hw, ww)
    strides = (*padded.strides[:-3],  # batch
               padded.strides[-2] * stride[0],  # H dimension
               padded.strides[-1] * stride[1],  # W dimension
               padded.strides[-3],  # input channel
               padded.strides[-2] * dilation[0],  # kernel height
               padded.strides[-1] * dilation[1],  # kernel width
               )

    expanded = xp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return expanded, padded


if fft_conv:
    # fft conv only support real numbers
    def _conv2d(xp, x, weight, stride, padding, dilation, groups, weight_dilated=None, no_cache=False):
        rfft2 = np_rfft2 if xp is np else cp_fft.rfft2
        irfft2 = np_irfft2 if xp is np else cp_fft.irfft2
        # dilate weight
        if weight_dilated is None:
            weight_dilated = _dilate(xp, weight, dilation, no_cache)
        # calculated target padding shape
        target_shape = _calc_pad_shape(x, weight_dilated, padding)
        # padding inpt and weight
        inpt_padded = _pad_right(xp, x, target_shape, padding)
        weight_padded = _pad_right(xp, xp.flip(weight_dilated, axis=(-1, -2)), target_shape, (0, 0))
        # compute fft
        inpt_hat = rfft2(inpt_padded)
        weight_hat = rfft2(weight_padded)
        # split n repeat
        weight_hat, weight_group_size = _strided_split(xp, weight_hat, -4, groups)
        inpt_hat, inpt_group_size = _strided_split(xp, inpt_hat, -3, groups)
        inpt_hat = _strided_repeat(xp, inpt_hat, -3, weight_group_size)
        # element wise mul. and sum
        y_hat = xp.squeeze(xp.moveaxis(inpt_hat[..., None], -4, -1) @ xp.moveaxis(weight_hat, -3, -1)[..., None],
                           (-1, -2))
        # merge group
        y_hat = y_hat.reshape(*y_hat.shape[:-4], -1, *y_hat.shape[-2:])
        # strip valid
        hp, wp = padding
        hi, wi = x.shape[-2:]
        hk, wk = weight_dilated.shape[-2:]
        y = irfft2(y_hat, s=target_shape)[..., (hk - 1):(hi + 2 * hp), (wk - 1):(wi + 2 * wp)]  # mode valid
        # stride
        hs, ws = stride
        value = y[..., ::hs, ::ws]
        return value.astype(x.dtype, copy=False), weight_dilated


    def _conv2d_backward_x(xp, grad, weight, stride, dilation, padding, groups, grad_dilated=None, weight_dilated=None,
                           x=None, output_padding=(0, 0), no_cache=False):
        rfft2 = np_rfft2 if xp is np else cp_fft.rfft2
        irfft2 = np_irfft2 if xp is np else cp_fft.irfft2
        # dilate
        if grad_dilated is None:
            grad_dilated = _dilate(xp, grad, stride, no_cache)
        if weight_dilated is None:
            weight_dilated = _dilate(xp, weight, dilation, no_cache)
        # calculated target padding shape
        target_shape = _calc_pad_shape(grad_dilated, weight_dilated, padding)
        # padding
        weight_padded = _pad_right(xp, weight_dilated, target_shape, (0, 0))
        grad_padded = _pad_right(xp, grad_dilated, target_shape, padding)
        # fft
        weight_hat = rfft2(weight_padded)
        grad_hat = rfft2(grad_padded)
        # split, repeat
        weight_hat, weight_group_size = _strided_split(xp, weight_hat, -4, groups)
        inpt_group_size = weight.shape[-3]
        grad_hat = _strided_repeat(xp, _strided_split(xp, grad_hat, -3, groups)[0], -2, inpt_group_size)
        # element-wise mul
        inpt_grad_hat = xp.squeeze(
            xp.moveaxis(grad_hat[..., None], -5, -1) @ xp.moveaxis(weight_hat, -4, -1)[..., None], (-1, -2))
        # merge
        inpt_grad_hat = inpt_grad_hat.reshape(*inpt_grad_hat.shape[:-4], -1, *inpt_grad_hat.shape[-2:])

        hg, wg = grad_dilated.shape[-2:]
        hk, wk = weight_dilated.shape[-2:]
        hp, wp = padding
        if x is not None:
            hi, wi = x.shape[-2:]
        else:
            hop = output_padding[0]
            wop = hop if len(output_padding) == 1 else output_padding[1]
            hi = (grad.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight.shape[-2] - 1) + hop + 1
            wi = (grad.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight.shape[-1] - 1) + wop + 1
        inpt_grad = irfft2(inpt_grad_hat, s=target_shape)[..., :(hg + hk + hp - 1), :(wg + wk + wp - 1)]  # mode full
        inpt_grad = inpt_grad[..., 2 * hp:(2 * hp + hi), 2 * wp:(2 * wp + wi)]
        inpt_grad = inpt_grad.astype(grad.dtype, copy=False)
        hig, wig = inpt_grad.shape[-2:]
        value = xp.pad(inpt_grad, ((0, 0), (0, 0), (0, hi - hig), (0, wi - wig)))
        return value, weight_dilated, grad_dilated


    def _conv2d_backward_w(xp, x, grad, weight, stride, dilation, padding, groups, grad_dilated=None):
        rfft2 = np_rfft2 if xp is np else cp_fft.rfft2
        irfft2 = np_irfft2 if xp is np else cp_fft.irfft2
        # dilated kernel
        if grad_dilated is None:
            grad_dilated = _dilate(xp, grad, stride)
        # calculated target padding shape
        target_shape = _calc_pad_shape(x, grad_dilated, padding)
        # padding
        inpt_padded = _pad_right(xp, x, target_shape, padding)
        grad_padded = _pad_right(xp, xp.flip(grad_dilated, (-1, -2)), target_shape, (0, 0))
        # fft
        inpt_hat = rfft2(inpt_padded)
        grad_hat = rfft2(grad_padded)
        # split, repeat
        inpt_hat, inpt_group_size = _strided_split(xp, inpt_hat, -3, groups)
        weight_group_size = weight.shape[-4] // groups
        inpt_hat = _strided_repeat(xp, inpt_hat, -3, weight_group_size)
        grad_hat = _strided_repeat(xp, _strided_split(xp, grad_hat, -3, groups)[0], -2, inpt_group_size)
        # element-wise mul. then sum along batch axis
        weight_grad_hat = xp.squeeze(xp.moveaxis(inpt_hat[..., None], 0, -1) @ xp.moveaxis(grad_hat, 0, -1)[..., None],
                                     (-1, -2))
        # merge group
        weight_grad_hat = weight_grad_hat.reshape(*weight_grad_hat.shape[:-5], -1, *weight_grad_hat.shape[-3:])
        hp, wp = padding
        hi, wi = x.shape[-2:]
        hk, wk = grad_dilated.shape[-2:]
        weight_grad = irfft2(weight_grad_hat, s=target_shape)[..., (hk - 1):(hi + 2 * hp),
                      (wk - 1):(wi + 2 * wp)]  # mode valid
        # stride
        hs, ws = dilation
        hk, wk = weight.shape[-2:]
        weight_grad = weight_grad[..., ::hs, ::ws]
        # select valid to be same dim as weight
        value = weight_grad[..., :hk, :wk]
        return value.astype(grad.dtype, copy=False), grad_dilated
else:
    def _conv2d(xp, x, weight, stride, padding, dilation, groups, backward_w=False, backward_x=False, no_cache=False):
        '''
        padding should be set to (0, 0) if x is already padded
        '''
        if xp is np:
            """
            if numpy is used, conv2d loop through group dimension
            """
            padded = _padding(xp, x, padding, (0, 0), False, 0, no_cache)
            if groups == 1:
                expanded = _sliding_window_view(xp, padded, weight.shape, stride, dilation)[0]
                if backward_x: weight = xp.swapaxes(weight, axis1=-3, axis2=-4)
                if backward_w:
                    expanded = xp.swapaxes(expanded, -3, 0)
                    weight = xp.swapaxes(weight, -3, 0)
                value = xp.einsum('NHWihw,oihw->NoHW', expanded, weight, optimize=True)
                if backward_w: value = xp.swapaxes(value, 1, 0)
            else:
                weight_split = _strided_split(xp, weight, -3 if backward_w else -4, groups)[0]
                padded_split = _strided_split(xp, padded, -3, groups)[0]
                results = []
                for g in range(groups):
                    p = padded_split[..., g, :, :, :]
                    w = weight_split[..., g, :, :, :] if backward_w else weight_split[g, ...]
                    results.append(
                        _conv2d(xp, p, w, stride, padding=(0, 0), dilation=dilation, groups=1, backward_w=backward_w,
                                backward_x=backward_x)[0])
                value = xp.concatenate(results, axis=-4 if backward_w else -3)
        else:
            """
            if cupy is used, conv2d is fully vectorized to take advantage of gpu
            """
            # pad input
            padded = _padding(xp, x, padding, (0, 0), False, 0, no_cache)
            # split
            if backward_w: weight = xp.swapaxes(weight, -3, -4)  # move contracted axes
            weight = _strided_split(xp, weight, -4, groups)[0]
            padded_split = _strided_split(xp, padded, -3, groups)[0]
            expanded = _sliding_window_view_groups(xp, padded_split, weight, stride, dilation)
            # move contracted axes
            if backward_x: weight = xp.swapaxes(weight, -3, -4)
            if backward_w: expanded = xp.swapaxes(expanded, -3, 0)
            # einsum
            value = xp.einsum('NHWgihw, goihw->NHWgo', expanded, weight, optimize=True)
            # merge the last two axes
            value = value.reshape((*value.shape[:-2], -1))
            # move axes for correct output shape
            value = xp.moveaxis(value, -1, -4 if backward_w else -3)
        return value, padded


    def _conv2d_backward_w(xp, x, grad, stride, padding, dilation, groups, weight):
        '''
        padding: set to (0, 0) if x is already padded
        '''
        x_padded = _padding(xp, x, padding)
        hw, ww = weight.shape[-2:]
        H_out, W_out = grad.shape[-2:]
        H_valid = (H_out - 1) * stride[0] + 1 + dilation[0] * (hw - 1)
        W_valid = (W_out - 1) * stride[1] + 1 + dilation[1] * (ww - 1)
        return _conv2d(xp, x_padded[..., :H_valid, :W_valid], grad, stride=dilation, padding=(0, 0), dilation=stride,
                       groups=groups, backward_w=True)


    def _conv2d_backward_x(xp, grad, weight, stride, padding, dilation, groups, x=None, output_padding=(0, 0)):
        ### dilate grad, then pad with (weight-1)*dilation to take into account full conv
        grad_dilated = _dilate(xp, grad, stride)  # dilate grad with stride
        weight_flipped = xp.flip(weight, axis=(-1, -2))  # flip weight
        ph, pw = padding
        dh, dw = dilation
        wh, ww = weight_flipped.shape[-2:]
        grad_padded = _padding(xp, grad_dilated, ((wh - 1) * dh, (ww - 1) * dw))  # pad grad_dilated

        x_grad = \
            _conv2d(xp, grad_padded, weight_flipped, stride=(1, 1), padding=(0, 0), dilation=dilation, groups=groups,
                    backward_x=True)[0]

        if x is not None:
            Hx, Wx = x.shape[-2:]
        else:
            hop = output_padding[0]
            wop = hop if len(output_padding) == 1 else output_padding[1]
            Hx = (grad.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight.shape[-2] - 1) + hop + 1
            Wx = (grad.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight.shape[-1] - 1) + wop + 1
        x_grad = x_grad[..., ph:Hx + ph, pw:Wx + pw]
        hig, wig = x_grad.shape[-2:]
        x_grad = xp.pad(x_grad, ((0, 0), (0, 0), (0, Hx - hig), (0, Wx - wig)))
        return x_grad


def conv2d(inpt, weight, bias, stride, padding, dilation, groups):
    """
    note that the output is not contiguous
    """
    x_data = inpt.data
    # if not x_data.flags['C_CONTIGUOUS']:
    #     warnings.warn(
    #         'Input to Conv2d is not contiguous, performance may drop significantly. Use x = x.contiguous()',
    #         RuntimeWarning)
    if x_data.ndim < 4:
        raise RuntimeError(f'Expected 4D (batched) input to conv2d, '
                           f'but got input of size: {x_data.shape}')

    if groups * weight.shape[-3] != inpt.shape[-3]:
        raise RuntimeError(
            'Given groups={}, weight of size {}, expected input{} to have {} channels, but got {} channels instead'.format(
                groups, weight.shape, inpt.shape, groups * weight.shape[-3], inpt.shape[-3]))

    xp = cp if x_data.__class__ is cp_ndarray else np
    weight_data = weight.data
    requires_grad = weight.requires_grad | inpt.requires_grad
    value, weight_dilated_or_x_padded = _conv2d(xp, x_data, weight_data, stride, padding, dilation, groups,
                                                no_cache=True)
    if bias is not None:
        requires_grad |= bias.requires_grad
        value += bias.data[:, None, None]
    output = build_links(value, requires_grad, conv2d, inpt, weight, bias, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, weight_dilated_or_x_padded=weight_dilated_or_x_padded)
    return output


@register_gradients(conv2d)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    stride = params['stride']
    padding = params['padding']
    dilation = params['dilation']
    groups = params['groups']

    inpt = inputs[0]  # input x
    weight = inputs[1]  # weight
    bias = inputs[2]  # bias, can be None

    grad_dilated = None

    if bias and bias.requires_grad:  # bias
        bias.grad += xp.einsum('Nohw->o', grad)

    if weight.requires_grad:  # weight
        if fft_conv:
            value, grad_dilated = _conv2d_backward_w(xp, inpt.data, grad, weight.data, stride, dilation, padding,
                                                     groups)
        else:
            x_padded = params['weight_dilated_or_x_padded']
            value = _conv2d_backward_w(xp, x_padded, grad, stride, (0, 0), dilation, groups, weight.data)[0]
        weight.grad += value

    if inpt.requires_grad:  # inpt
        if fft_conv:
            weight_dilated = params['weight_dilated_or_x_padded']
            value = \
            _conv2d_backward_x(xp, grad, weight.data, stride, dilation, padding, groups, grad_dilated, weight_dilated,
                               inpt.data)[0]
        else:
            value = _conv2d_backward_x(xp, grad, weight.data, stride, padding, dilation, groups, x=inpt.data)
        inpt.grad += value


def conv_transpose2d(inpt, weight, bias, stride, padding, output_padding, groups, dilation):
    """
    note that the output is not contiguous
    """
    x_data = inpt.data
    # if not x_data.flags['C_CONTIGUOUS']:
    #     warnings.warn(
    #         'Input to ConvTranspose2d is not contiguous, performance may drop significantly. Use x = x.contiguous()',
    #         RuntimeWarning)
    if x_data.ndim < 4:
        raise RuntimeError(f'Expected 4D (batched) input to conv_transpose2d, '
                           f'but got input of size: {x_data.shape}')

    if weight.shape[-4] != inpt.shape[-3]:
        raise RuntimeError(
            'Given transposed=1, weight of size {}, expected input {} to have {} channels, but got {} channels instead'.format(
                weight.shape, inpt.shape, weight.shape[-4], inpt.shape[-3]))

    xp = cp if x_data.__class__ is cp_ndarray else np
    weight_data = weight.data
    requires_grad = weight.requires_grad | inpt.requires_grad
    if fft_conv:
        value, weight_dilated, grad_dilated = _conv2d_backward_x(xp, x_data, weight_data, stride, dilation, padding,
                                                                 groups, output_padding=output_padding, no_cache=True)
        if bias is not None:
            requires_grad |= bias.requires_grad
            value += bias.data[:, None, None]

        output = build_links(value, requires_grad, conv_transpose2d, weight, bias, inpt, stride=stride, padding=padding,
                             dilation=dilation, groups=groups, weight_dilated=weight_dilated, grad_dilated=grad_dilated)
    else:
        value = _conv2d_backward_x(xp, x_data, weight_data, stride, padding, dilation, groups,
                                   output_padding=output_padding)
        if bias is not None:
            requires_grad |= bias.requires_grad
            value += bias.data[:, None, None]

        output = build_links(value, requires_grad, conv_transpose2d, inpt, weight, bias, stride=stride, padding=padding,
                             dilation=dilation, groups=groups)
    return output


@register_gradients(conv_transpose2d)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    stride = params['stride']
    padding = params['padding']
    dilation = params['dilation']
    groups = params['groups']

    inpt = inputs[0]  # input x
    weight = inputs[1]  # weight
    bias = inputs[2]  # bias, can be None

    grad_padded = None

    if bias and bias.requires_grad:  # bias
        bias.grad += xp.einsum('Nohw->o', grad)

    if weight.requires_grad:  # weight
        if fft_conv:
            grad_dilated = params['grad_dilated']
            value = \
                _conv2d_backward_w(xp, grad, inpt.data, weight.data, stride, dilation, padding, groups, grad_dilated)[0]
        else:
            value, grad_padded = _conv2d_backward_w(xp, grad, inpt.data, stride, padding, dilation, groups, weight.data)
        weight.grad += value

    if inpt.requires_grad:  # inpt
        if fft_conv:
            weight_dilated = params['weight_dilated']
            value = _conv2d(xp, grad, weight.data, stride, padding, dilation, groups, weight_dilated)[0]
        else:
            value = \
            _conv2d(xp, grad if grad_padded is None else grad_padded, weight.data, stride, (0, 0), dilation, groups)[0]
        inpt.grad += value


def _max_pool2d(xp, x, kernel_size, stride, padding, dilation, ceil_mode):
    low_dim_flag = False
    if x.ndim == 3:
        low_dim_flag = True
        x = x[None, ...]
    expanded = _sliding_window_view(xp, x, kernel_size, stride, dilation, padding, ceil_mode, val=xp.NINF)[0]
    idx1 = xp.nanargmax(expanded, -1)
    # equivalent to xp.indices(expanded.shape[:-1], sparse=True) but faster
    ax0 = xp.arange(expanded.shape[0])[:, None, None, None, None]
    ax1 = xp.arange(expanded.shape[1])[None, :, None, None, None]
    ax2 = xp.arange(expanded.shape[2])[None, None, :, None, None]
    ax3 = xp.arange(expanded.shape[3])[None, None, None, :, None]
    ax4 = xp.arange(expanded.shape[4])[None, None, None, None, :]
    value = expanded[ax0, ax1, ax2, ax3, ax4, idx1]
    idx2 = xp.nanargmax(value, -1)
    ax0 = ax0[..., 0]
    ax1 = ax1[..., 0]
    ax2 = ax2[..., 0]
    ax3 = ax3[..., 0]
    value = value[ax0, ax1, ax2, ax3, idx2]
    value = xp.moveaxis(value, -1, -3)
    pos = (ax0, ax1, ax2, ax3, idx1, idx2)
    if low_dim_flag:
        value = value[0]
    return value, pos


def _max_pool2d_backward(xp, x, grad, pos, kernel_size, stride, padding, dilation, ceil_mode):
    x_grad = xp.zeros_like(x)
    ax0, ax1, ax2, ax3, idx1, idx2 = pos
    expanded, padded = _sliding_window_view(xp, x_grad, kernel_size, stride, dilation, padding, ceil_mode, val=xp.NINF)
    idx1_m = idx1[ax0, ax1, ax2, ax3, idx2]
    expanded[ax0, ax1, ax2, ax3, idx2, idx1_m] = xp.moveaxis(grad, -3, -1)
    hp, wp = padding
    h, w = x_grad.shape[-2:]
    x_grad = padded[..., hp:(hp + h), wp:(wp + w)]
    return x_grad


def max_pool2d(inpt, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    if padding[0] * 2 > kernel_size[0] or padding[1] * 2 > kernel_size[1]:
        raise RuntimeError(
            f'pad should be smaller than or equal to half of kernel size, but got padW = {padding[1]}, padH = {padding[0]}, kW = {kernel_size[1]}, kH = {kernel_size[0]}')
    if inpt.ndim != 3 and inpt.ndim != 4:
        raise RuntimeError('non-empty 3D or 4D (batch mode) tensor expected for input')
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    value, pos = _max_pool2d(xp, inpt.data, kernel_size, stride, padding, dilation, ceil_mode)
    output = build_links(value, inpt.requires_grad, max_pool2d, inpt, pos=pos, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, ceil_mode=ceil_mode)
    return (output, pos) if return_indices else output


@register_gradients(max_pool2d)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    pos = params['pos']
    kernel_size = params['kernel_size']
    stride = params['stride']
    padding = params['padding']
    dilation = params['dilation']
    ceil_mode = params['ceil_mode']
    x = inputs[0]
    if x.requires_grad:
        x.grad += _max_pool2d_backward(xp, x.data, grad, pos, kernel_size, stride, padding, dilation, ceil_mode)


def dropout(inpt: Tensor, p=0.5, training=True, inplace=False):
    """
    inplace can be true as long as tensor data is not used further during forward and not used during backward
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    if training and p > 0.0:  # if not training or if p == 0., return the incoming tensor as it is.
        mask = xp.random.binomial(1, 1 - p, size=x.shape)
        if inplace:
            x *= mask
            if p != 1.:
                x /= 1 - p
            value = x
        else:
            value = xp.multiply(x, mask, dtype=x.dtype)
            if p != 1.:
                value /= 1 - p
        return build_links(value, inpt.requires_grad, dropout, inpt, p=p, mask=mask)
    else:
        return inpt


@register_gradients(dropout)
def backward(tensor: Tensor, grad, params):
    inputs = tensor.parents
    p = params['p']
    mask = params['mask']
    x = inputs[0]
    xp = cp if grad.__class__ is cp_ndarray else np
    if x.requires_grad:
        x.grad += xp.multiply(grad, mask, dtype=grad.dtype) / (1 - p * (p != 1.))


def _verify_batch_size(size):
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    if size_prods == 1:
        raise ValueError("Expected more than 1 value per channel when training, got input size {}".format(size))


def batch_norm(inpt, running_mean, running_var, weight, bias, training, momentum, eps):
    def expand_dim(*arrays):
        # expand weight, bias running mean and var to (1,C,extra 1's), where number of extra 1's
        # is length of input dim-2
        extra_dim = x.ndim - 2
        return tuple(i.reshape(i.shape + (1,) * extra_dim) for i in arrays)

    # expand_dim(running_mean, running_var, weight, bias)
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    axis = (0,) + tuple(range(2, x.ndim))
    shape = (x.shape[0],) + x.shape[2:]
    if training:
        _verify_batch_size(inpt.shape)
        batch_mean = xp.mean(x, axis=axis)
        batch_var = xp.var(x, axis=axis)
        batch_sd = xp.sqrt(batch_var + eps)
        if running_mean is not None and running_var is not None:
            # https://www.geeksforgeeks.org/python-multiply-numbers-list-3-different-ways/
            N = prod(shape)
            sample_var = xp.multiply(batch_var, N / (N - 1), dtype=inpt.dtype)
            running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean
            running_var.data = (1 - momentum) * running_var.data + momentum * sample_var
    else:
        if running_mean is not None and running_var is not None:
            batch_mean = running_mean.data
            batch_var = running_var.data
            batch_sd = xp.sqrt(batch_var + eps)
        else:
            batch_mean = xp.mean(x, axis=axis)
            batch_var = xp.var(x, axis=axis)
            batch_sd = xp.sqrt(batch_var + eps)

    requires_grad = inpt.requires_grad
    batch_mean, batch_var, batch_sd = expand_dim(batch_mean, batch_var, batch_sd)
    norm = (x - batch_mean) / batch_sd
    # affine
    weight_data = None
    value = norm
    if weight is not None:
        weight_data = expand_dim(weight.data)[0]
        value *= weight_data
        requires_grad |= weight.requires_grad
    if bias is not None:
        bias_data = expand_dim(bias.data)[0]
        value += bias_data
        requires_grad |= bias.requires_grad

    output = build_links(value, requires_grad, batch_norm, inpt, weight, bias, axis=axis, shape=shape, norm=norm,
                         mean=batch_mean, var=batch_var + eps, sd=batch_sd, weight_data=weight_data)
    return output


def layer_norm(inpt, normalized_shape, weight, bias, eps):
    if weight is not None:
        if weight.shape != normalized_shape:
            raise RuntimeError(
                f'Expected weight to be of same shape as normalized_shape, but got weight of shape {weight.shape} and normalized_shape = {normalized_shape}')
    if bias is not None:
        if bias.shape != normalized_shape:
            raise RuntimeError(
                f'Expected bias to be of same shape as normalized_shape, but got bias of shape {bias.shape} and normalized_shape = {normalized_shape}')
    if inpt.shape[-len(normalized_shape):] != normalized_shape:
        raise RuntimeError(
            f'Given normalized_shape={normalized_shape}, expected input with shape (*, {", ".join([str(_) for _ in normalized_shape])}), but got input of size{inpt.shape}')
    x = inpt.data
    xp = cp if x.__class__ is cp_ndarray else np
    axis = tuple(-i for i in range(1, 1 + len(normalized_shape)))
    shape = x.shape[-len(normalized_shape):]
    layer_mean = xp.mean(x, axis=axis, keepdims=True)
    layer_var = xp.var(x, axis=axis, keepdims=True)
    layer_sd = xp.sqrt(layer_var + eps)

    requires_grad = inpt.requires_grad
    norm = (x - layer_mean) / layer_sd
    # elementwise affine
    weight_data = None
    value = norm
    if weight is not None:
        weight_data = weight.data
        value *= weight_data
        requires_grad |= weight.requires_grad
    if bias is not None:
        value += bias.data
        requires_grad |= bias.requires_grad
    output = build_links(value, requires_grad, layer_norm, inpt, weight, bias, axis=axis, shape=shape, norm=norm,
                         mean=layer_mean, var=layer_var + eps, sd=layer_sd, weight_data=weight_data)
    return output


@register_gradients(batch_norm, layer_norm)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    axis = params['axis']
    shape = params['shape']
    norm = params['norm']
    mu = params['mean']
    var = params['var']
    sd = params['sd']

    x = inputs[0]  # input
    weight = inputs[1]  # weight, can be None
    bias = inputs[2]  # bias, can be None

    sum_axis = axis if tensor.grad_fn is batch_norm else tuple(range(grad.ndim - len(axis)))

    if weight and weight.requires_grad:
        weight.grad += (norm * grad).sum(sum_axis)
    if bias and bias.requires_grad:
        bias.grad += grad.sum(sum_axis)
    if x.requires_grad:
        if weight:
            weight_data = params['weight_data']
            grad = grad * weight_data
        N = prod(shape)
        a = xp.multiply(N, grad, dtype=grad.dtype) - grad.sum(axis=axis, keepdims=True)
        b = (x.data - mu) / var * xp.sum(grad * (x.data - mu), axis=axis, keepdims=True)
        value = xp.divide(a - b, N, dtype=grad.dtype) / sd
        x.grad += value


def embedding(inpt, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    if not isinstance(inpt, Tensor):
        raise TypeError(f'embedding(): input must be Tensor, not {inpt.__class__.__name__}')
    if not sparse_is_loaded and sparse:
        raise RuntimeError('Numpy do not support sparse, set sprse=False, or install cupy/scipy')
    # change padding index to positive
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.shape[0], "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.shape[0], "Padding_idx must be within num_embeddings"
            padding_idx = weight.shape[0] + padding_idx
    else:
        padding_idx = -1

    xp = cp if inpt.data.__class__ is cp_ndarray else np
    # renorm if max_norm is not None
    indices, counts = None, None
    if max_norm is not None:
        inpt = inpt.contiguous()
        indices, counts = xp.unique(inpt.data, return_counts=True)
        norm = xp.linalg.norm(weight.data[indices], ord=norm_type, axis=-1, keepdims=True)
        rescale = (norm > max_norm)[:, 0]
        selection = xp.zeros(weight.shape[0], dtype=bool)
        selection[indices] = rescale
        weight.data[selection] *= max_norm / (norm[rescale] + 1e-7)
    value = weight.data[inpt.data]
    return build_links(value, weight.requires_grad, embedding, weight, inpt=inpt, padding_idx=padding_idx,
                       scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, indices=indices, counts=counts)


@register_gradients(embedding)
def backward(tensor: Tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inpt = params['inpt']
    padding_idx = params['padding_idx']  # can be None
    scale_grad_by_freq = params['scale_grad_by_freq']  # bool
    sparse = params['sparse']  # bool
    indices = params['indices']
    counts = params['counts']
    W = tensor.parents[0]
    if W.requires_grad:
        inpt = inpt.data
        weight = W.data
        inpt_flatten = inpt.flatten()
        grad_reshaped = grad.reshape(-1, grad.shape[-1])  # to 2d matrix
        if sparse:
            csr_matrix = cp_sparse.csr_matrix if xp is cp else sci_sparse.csr_matrix
            # argsort input and grad
            p = inpt_flatten.argsort()
            grad_reshaped = grad_reshaped[p]
            inpt_flatten = inpt_flatten[p]
            # input frequency count
            bin_count = xp.bincount(inpt_flatten + 1, minlength=weight.shape[-2] + 1)
            if scale_grad_by_freq:
                grad_reshaped /= bin_count[inpt_flatten + 1][
                    ..., None]  # inplace division, float32 /= float64 won't change dtyep
            # construct sparse grad
            data = grad_reshaped.flatten()
            ind = xp.tile(xp.arange(weight.shape[-1]), len(inpt_flatten))
            indptr = (weight.shape[-1] * bin_count).cumsum()
            value = csr_matrix((data, ind, indptr), shape=weight.shape)
            # pytorch doesn't sum duplicates, no need for value.sum_duplicates()
            # padding index
            if padding_idx >= 0:
                value.data[value.indptr[padding_idx]:value.indptr[padding_idx + 1]] = 0
        else:
            scatter_add = cpx.scatter_add if xp is cp else np.add.at
            value = xp.zeros_like(weight)
            scatter_add(value, inpt_flatten, grad_reshaped)
            if padding_idx >= 0:
                value[padding_idx] = 0
            if scale_grad_by_freq:
                if indices is None:
                    indices, counts = xp.unique(inpt.data, return_counts=True)
                value[indices] /= counts[:, None]
        W.grad += xp.array(value)


def _in_projection_packed(q, k, v, w, b=None):
    """
    shape:
    - q: (tgt, bsz, E)
    - k: (src, bsz, E)
    - v: (src1, bsz, E) # may be different than k
    - w: (E * 3, E)
    - b: (E * 3)
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)

        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None):
    """
    - q: (Qdims..., Eq), k: (Kdims..., Ek), v: (Vdims..., Ev)
    where Eq,Ek,Ev are the query,key,value embedding dimensions and Qdims,Kdims,Vdims are any
    number of leading dimensions.
    - w_q: (Eq, Eq), w_k: (Eq, Ek), w_v: (Eq, Ev)
    - b_q: (Eq), b_k: (Eq), b_v: (Eq)

    Output:
     - q':(Qdims..., Eq), k':(Kdims..., Eq), v':(Vdims..., Eq)
    """
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(q, k, v, attn_mask=None, training=False, dropout_p=0.0):
    bsz, tgt, E = q.shape
    q = q / math.sqrt(E)
    # (bsz, tgt, E) @ (bsz, E, src) -> (bsz, tgt, src)
    attn = matmul(q, k.swapaxes(-2, -1))
    # attention mask
    if attn_mask is not None:
        attn += attn_mask
    # softmax
    attn = softmax(attn, -1)
    # dropout
    if dropout_p > 0.0:
        attn = dropout(attn, training=training, p=dropout_p)
    # (bsz, tgt, src) @ (bsz, src, E) -> (bsz, tgt, E)
    output = matmul(attn, v)
    return output, attn


def _mha_precheck(query, key, value, key_padding_mask, attn_mask, num_heads, embed_dim_to_check,
                  use_separate_proj_weight, in_proj_bias, q_proj_weight, k_proj_weight, v_proj_weight, bias_k, bias_v):
    tgt_len, bsz, Eq = query.shape
    src_len, _, Ek = key.shape
    _, _, Ev = value.shape

    # shape check
    if query.dim() == 3:
        # Batched Inputs
        assert key.dim() == 3 and value.dim() == 3, ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
                                                     f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() == 3, ("For batched (3-D) `query`, expected `attn_mask` to be `None` or 3-D"
                                          f" but found {attn_mask.dim()}-D tensor instead")
            assert attn_mask.dtype.type in _floating_point or attn_mask.dtype == bool, f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
    else:
        raise AssertionError(f"query should be batched 3D tensor but received {query.dim()}-D query tensor")

    assert Eq == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {Eq}"
    assert Eq % num_heads == 0, f"embed_dim {Eq} not divisible by num_heads {num_heads}"

    if use_separate_proj_weight:
        # allow Multi Head Attention to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[
                                :2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        assert q_proj_weight.shape == (
            Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {q_proj_weight.shape}"
        assert k_proj_weight.shape == (
            Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {k_proj_weight.shape}"
        assert v_proj_weight.shape == (
            Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {v_proj_weight.shape}"

        if in_proj_bias is not None:
            separate_bias_shape = in_proj_bias.shape[0] // 3
            assert separate_bias_shape == Eq, f"expecting query, key and value bias shape of {(Eq,)}, but got {(separate_bias_shape,)}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    assert (bias_k is None) == (bias_v is None), 'bias_k and bias_v should either both existed or both be None.'


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k,
                                 bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True,
                                 key_padding_mask=None, need_weights=True, attn_mask=None,
                                 use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                 v_proj_weight=None, average_attn_weights=True):
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads

    # precheck
    _mha_precheck(query, key, value, key_padding_mask, attn_mask, num_heads, embed_dim_to_check,
                  use_separate_proj_weight, in_proj_bias, q_proj_weight, k_proj_weight, v_proj_weight, bias_k, bias_v)

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # add bias along batch dimension (currently second)
    # bias_k and bias_v shape: (1, 1, embed_dim)
    if bias_k is not None and bias_v is not None:
        k = cat([k, bias_k.repeat(1, bsz, 1)])  # k shape: cat([ (src, bsz, emb), (1, bsz, emb) ])
        v = cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:  # pad last dimension: left 0 right 1
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    # q shape: (tgt,bsz,emb)->(tgt,bsz*num_head,head_dim)->(bsz*num_head,tgt,head_dim)
    q = q.contiguous().view((tgt_len, bsz * num_heads, head_dim)).swapaxes(0, 1)
    k = k.contiguous().view((k.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    v = v.contiguous().view((v.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        # (bsz*num_head,src,head_dim)->(bsz*num_head,src+1,head_dim)
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        xp = cp if k.data.__class__ is cp_ndarray else np
        k = cat([k, Tensor(xp.zeros(zero_attn_shape, dtype=k.dtype), copy=False)], dim=1)
        v = cat([v, Tensor(xp.zeros(zero_attn_shape, dtype=v.dtype), copy=False)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz, src_len), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"

        key_padding_mask = key_padding_mask.view((bsz, 1, 1, src_len)).expand(-1, num_heads, -1, -1).reshape(
            bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == bool:
        new_attn_mask = zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    #
    # calculate attention and out projection
    #
    # attn_output: (bsz*num_head, tgt, E); attn_output_weights: (bsz*num_head, tgt, src)
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, training, dropout_p)
    # (bsz*num_head,tgt,head)->(tgt,bsz*num_head,head)->(tgt,bsz,head*num_head)==(tgt,bsz,embed_dim)
    attn_output = attn_output.swapaxes(0, 1).contiguous().view((tgt_len, bsz, embed_dim))
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(axis=1, keepdims=False) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None
