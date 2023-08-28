import math
from tortto import *

"""
datatype inconsistencies in cupy:
1. a small float32 / large int will result in float64:
    x=cp.array([1,2,3]).astype('f')
    y=x/123456
    y.dtype --> float64
"""

scipy_is_loaded = bool(find_spec('scipy'))
sparse_is_loaded = cupy_is_loaded or scipy_is_loaded
if cupy_is_loaded:
    import cupyx as cpx
    import cupyx.scipy.special as cp_special
    import cupyx.scipy.sparse as cp_sparse

SQRT_PI={float16: float16(math.sqrt(math.pi)), float32: float32(math.sqrt(math.pi)), float64: float64(math.sqrt(math.pi))}
SQRT_2 = {float16: float16(math.sqrt(2)), float32: float32(math.sqrt(2)), float64: float64(math.sqrt(2))}
SQRT_2_over_PI = {float16: float16(math.sqrt(2 / math.pi)), float32: float32(math.sqrt(2 / math.pi)),
                  float64: float64(math.sqrt(2 / math.pi))}

# if scipy is present, replace default numpy functions to scipy
if scipy_is_loaded:
    import scipy.special as scipy_special
    import scipy.sparse as sci_sparse

# math has prod since python 3.8, math.prod is 30 times faster
prod = math.prod if hasattr(math, 'prod') else np.prod


class Relu(Function):
    """
    import cupy as cp
    from cupyx.profiler import benchmark

    def relu(x):
        return x * (x > 0)
    def relu_1(x):
        return cp.maximum(x, 0)
    x = cp.random.random((128, 1024, 16, 16)).astype('f')

    print(benchmark(relu, (x,), n_repeat=2000))
    # relu:CPU:46.473 us+/-8.977 (min:41.558/max:181.217)us GPU-0:1234.155us+/-19.290(min:1200.064/max:1705.344)us

    print(benchmark(relu_1, (x,), n_repeat=2000))
    # relu_1:CPU:23.442us+/-7.013(min:19.443/max:129.968)us GPU-0:711.736us+/-13.863(min:690.176/max:960.096)us
    """
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        inplace = params['inplace']
        xp = ctx.xp
        if inplace:
            inplace_precheck(xt0)
            xp.maximum(xd0, 0, out=xd0) # maximum propagates NaNs, fmax doesn't
            yt0 = inplace_update(xt0, ctx)
        else:
            yt0=build_links(xp.maximum(xd0, 0), grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * (yd0 > 0)
        return grad0

class LeakyRelu(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        inplace = params['inplace']
        negative_slope=params['negative_slope']
        xp = ctx.xp
        if inplace:
            inplace_precheck(xt0)
            xp.maximum(xd0 * negative_slope, xd0, out=xd0)
            yt0=inplace_update(xt0, ctx)
        else:
            yt0 = build_links(xp.maximum(xd0 * negative_slope, xd0), grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        negative_slope = ctx.params['negative_slope']
        gd0[xd0<0] *= negative_slope
        return gd0

class Gelu(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        approximate = params['approximate']
        xp = ctx.xp
        dtype = xd0.dtype.type
        if approximate == 'none':
            if not scipy_is_loaded and xp is np: # if scipy is not installed, and x is in cpu
                raise RuntimeError(f"SciPy is not installed, can't use approximate='none', set it to 'tanh' instead.")
            erf_s = cp_special.erf(xd0/SQRT_2[dtype]) if xp is cp else scipy_special.erf(xd0/SQRT_2[dtype])
        elif approximate == 'tanh':
            erf_s = xp.tanh(SQRT_2_over_PI[dtype] * (xd0 + .044715 * xd0 * xd0 * xd0))
        else:
            raise RuntimeError(f"approximate argument must be either none or tanh.")
        yt0 = build_links(0.5 * xd0 * (erf_s + 1), grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        approximate = ctx.params['approximate']
        xp = ctx.xp
        dtype = xd0.dtype.type
        s = xd0 / SQRT_2[dtype]
        if approximate == 'none':
            erf_s = cp_special.erf(s) if xp is cp else scipy_special.erf(s)
        else:
            erf_s = xp.tanh(SQRT_2_over_PI[dtype] * (xd0 + .044715 * xd0 * xd0 * xd0))
        erf_p = 2/SQRT_PI[dtype] * xp.exp(-(s * s))
        grad0 = gd0 * (0.5 + 0.5 * (erf_s + (xd0 * erf_p) / SQRT_2[dtype]))
        return grad0

class MseLoss(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs # input, target
        xd0, xd1 = xt0.data, xt1.data
        reduction=params['reduction']
        if xd0.shape != xd1.shape:
            warnings.warn(f"Using a target size ({xd1.shape}) that is different to the input size ({xd0.shape}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.", stacklevel=2)
        yd0 = (xd0 - xd1) ** 2
        if reduction == 'mean':
            yd0 = yd0.mean()  # note: loss is now a numpy array not Tensor
        elif reduction == 'sum':
            yd0 = yd0.sum()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        import tortto as tt
        x=tt.nparray(1.,dtype=tt.cp.float32)
        y=x*x.dtype.type(3)
        y1=x.dtype.type(3)*x
        print(y.__class__)
        print(y1.__class__)
        """
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        reduction = ctx.params['reduction']
        grad0 = gd0 * (xd0-xd1) * 2
        if reduction == 'mean':
            grad0 /= xd0.size
        return grad0 if ctx.needs_input_grad[0] else None, -grad0 if ctx.needs_input_grad[1] else None



class BinaryCrossEntropy(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs # input, target
        xd0, xd1 = xt0.data, xt1.data
        reduction = params['reduction']
        weight = params['weight']
        if xd0.shape != xd1.shape:
            warnings.warn(f"Using a target size ({xd1.shape}) that is different to the input size ({xd0.shape}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.", stacklevel=2)
        xp = ctx.xp
        yd0 = -(xd1 * xp.clip(xp.log(xd0), -100, None) + (1 - xd1) * xp.clip(xp.log(1 - xd0), -100, None))
        if weight is not None:
            yd0*=weight.data
        if reduction == 'mean':
            yd0 = yd0.mean()
        elif reduction == 'sum':
            yd0 = yd0.sum()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1, weight)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        reduction = ctx.params['reduction']
        xd0, xd1, weight = ctx.saved_tensors
        xp = ctx.xp
        common = gd0 if weight is None else gd0 * weight.data
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = common * (xd0-xd1) * xp.clip(1 / xd0, None, 1e12) * xp.clip(1 / (1-xd0), None, 1e12)
            if reduction == 'mean':
                grad0 /= xd0.size
        if ctx.needs_input_grad[1]:
            grad1 = common * xp.log(1 / xd0 - 1)
            if reduction == 'mean':
                grad1 /= xd1.size
        return grad0, grad1

class BinaryCrossEntropyWithLogits(Function):
    """
    import cupy as cp
    from cupyx.profiler import benchmark

    def logsig(x):
        return x * (x < 0) - cp.log1p(cp.exp(-cp.abs(x)))
    def logsig_1(x):
        return -cp.logaddexp(0,-x)
    x = cp.random.random((128, 512, 16, 16)).astype('f')

    print(benchmark(logsig, (x,), n_repeat=200))
    # logsig: CPU:101.254us+/-14.007(min:94.604/max:228.097)us GPU-0:2532.766us+/-45.736(min:2510.592/max:2865.440)us
    print(benchmark(logsig_1, (x,), n_repeat=200))
    # logsig_1: CPU:43.562us+/-3.755(min:41.323/max:75.371)us GPU-0:1125.976us+/-7.451(min:1122.080/max:1166.752)us
    """
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs  # input, target
        xd0, xd1 = xt0.data, xt1.data
        reduction = params['reduction']
        weight = params['weight']
        pos_weight = params['pos_weight']
        if xd0.shape != xd1.shape:
            raise ValueError(f"Target size ({xd1.shape}) must be the same as input size ({xd0.shape})")
        xp = ctx.xp
        log_sigmoid=xp.logaddexp(0, -xd0)
        if pos_weight is None:
            yd0 = log_sigmoid + xd0 - xd0 * xd1
        else:
            yd0 = (1 - xd1 * (1 - pos_weight.data)) * log_sigmoid + xd0 - xd0 * xd1
        if weight is not None:
            yd0*=weight.data
        if reduction == 'mean':
            yd0 = yd0.mean()  # note: loss is now a numpy array not Tensor
        elif reduction == 'sum':
            yd0 = yd0.sum()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1, weight, pos_weight)
        ctx.params['log_sigmoid']=log_sigmoid
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        reduction = ctx.params['reduction']
        log_sigmoid = ctx.params['log_sigmoid']
        xd0, xd1, weight, pos_weight = ctx.saved_tensors
        xp = ctx.xp
        common = gd0 if weight is None else gd0 * weight.data
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            if pos_weight is None:
                grad0 = common * (xp.exp(-log_sigmoid)-xd1)
            else:
                grad0 = common * ((xd1*(1-pos_weight.data)-1)*(1-xp.exp(-log_sigmoid))+1-xd1)
            if reduction == 'mean':
                grad0 /= xd0.size
        if ctx.needs_input_grad[1]:
            if pos_weight is None:
                grad1 = common * -xd0
            else:
                grad1 = common * ((pos_weight.data-1)*log_sigmoid-xd0)

            if reduction == 'mean':
                grad1 /= xd1.size
        return grad0, grad1

class NllLoss(Function): # input has ndim > 1
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xt1 = params['target']
        xd0, xd1 = xt0.data, xt1.data
        reduction = params['reduction']
        weight = params['weight']
        ignore_index = params['ignore_index']
        xp = ctx.xp
        # weight
        w=None
        if weight is not None:
            w=weight.data[xd1]
            if ignore_index>=0:
                w *= xd1 != ignore_index
        else:
            if ignore_index>=0:
                w = xd1 != ignore_index # w is int64
        idx = np.indices(xd1.shape, sparse=True)
        criteria = xd1 if len(idx)==0 else (idx[0], xd1, *idx[1:])
        yd0 = -xd0[criteria]
        if w is not None:
            yd0*=w
        if reduction == 'sum':
            yd0 = yd0.sum()
        elif reduction == 'mean':
            if w is None:
                N=yd0.size
                yd0=yd0.mean()
            else:
                N = xp.count_nonzero(w)
                yd0=xp.divide(yd0.sum(), N, dtype=xd0.dtype)
            ctx.params['N']=N
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1, weight)
        ctx.params['criteria'] = criteria
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        reduction = ctx.params['reduction']
        ignore_index = ctx.params['ignore_index']
        criteria = ctx.params['criteria']
        xd0, xd1, w = ctx.saved_tensors
        # weight
        if w is not None:
            w = w[xd1]
            if ignore_index >= 0:
                w *= xd1 != ignore_index
        else:
            if ignore_index >= 0:
                w = xd1 != ignore_index
        if reduction == 'mean':
            gd0/=ctx.params['N']
        xp = ctx.xp
        grad0=xp.zeros_like(xd0)
        grad0[criteria]=-gd0 if w is None else -gd0*w
        return grad0

class Softmax(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        aug = xd0 - xp.max(xd0, axis=dim, keepdims=True)
        exp = xp.exp(aug)
        sum_exp = xp.sum(exp, axis=dim, keepdims=True)
        yt0 = build_links(exp / sum_exp, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim=ctx.params['dim']
        yd0,=ctx.saved_tensors
        grad0=(gd0 - (gd0 * yd0).sum(dim, keepdims=True)) * yd0
        return grad0


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        aug = xd0 - xp.max(xd0, axis=dim, keepdims=True)
        log_sum_exp = xp.log(xp.sum(xp.exp(aug), axis=dim, keepdims=True))
        yt0 = build_links(aug-log_sum_exp, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim = ctx.params['dim']
        yd0, = ctx.saved_tensors
        xp = ctx.xp
        grad0 = gd0 - gd0.sum(axis=dim, keepdims=True) * xp.exp(yd0)
        return grad0

class LogSigmoid(Function):
    # different from pytorch. this function saves both xt0 and yt0, whereas pytorch only saves xt0
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        yt0 = build_links(-xp.logaddexp(0,-xd0), grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0,= ctx.saved_tensors
        xp = ctx.xp
        grad0 = gd0 * xp.exp(-xp.logaddexp(0,xd0))
        return grad0


class ConstantPad(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        pad=params['pad']
        value=params['value']
        padding_dims = len(pad) // 2  # number of dimensions that needs padding
        pad_width=tuple((0, 0) if i >= padding_dims else pad[(2 * i):(2 * i + 2)] for i in reversed(range(xd0.ndim)))
        xp = ctx.xp
        yt0 = build_links(xp.pad(xd0, pad_width, mode='constant', constant_values=value), grad_fn=ctx)
        ctx.params={'original_shape': xd0.shape, 'pad_width':pad_width}
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        original_shape = ctx.params['original_shape']
        pad_width = ctx.params['pad_width']
        selection = tuple(slice(pad_width[i][0], pad_width[i][0] + original_shape[i]) for i in range(len(original_shape)))
        grad0 = gd0[selection]
        return grad0

########### Conv2d and ConvTransposed2d

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
    # dilate grad, then pad with (weight-1)*dilation to take into account full conv
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

class Convolution(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs # input, weight, bias
        xd0, xd1 = xt0.data, xt1.data
        xd2=None if xt2 is None else xt2.data
        stride=params['stride']
        padding=params['padding']
        dilation=params['dilation']
        groups=params['groups']
        # need to make sure input and weight are on same device,
        # because np.einsum can take a numpy and a cupy array as inputs and outputs a cupy array
        if xd0.__class__ is not xd1.__class__:
            raise RuntimeError(f'Input type ({xd0.__class__.__name__}) and weight type ({xd1.__class__.__name__}) should be the same')
        if xd2 is not None and xd0.__class__ is not xd2.__class__:
            raise RuntimeError(f'Input type ({xd0.__class__.__name__}) and bias type ({xd2.__class__.__name__}) should be the same')
        if xd0.ndim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv2d, '
                               f'but got input of size: {xd0.shape}')
        if groups * xd1.shape[-3] != xd0.shape[-3]:
            raise RuntimeError(f'Given groups={groups}, weight of size {xd1.shape}, '
                               f'expected input{xd0.shape} to have {groups * xd1.shape[-3]} channels, '
                               f'but got {xd0.shape[-3]} channels instead')

        xp = ctx.xp
        yd0, x_padded = _conv2d(xp, xd0, xd1, stride, padding, dilation, groups, no_cache=True)
        if xd2 is not None:
            yd0 += xd2[:, None, None]
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        ctx.params['x_padded'] = x_padded
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1= ctx.saved_tensors
        stride = ctx.params['stride']
        padding = ctx.params['padding']
        dilation = ctx.params['dilation']
        groups = ctx.params['groups']
        xp = ctx.xp
        grad0, grad1, grad2=None, None, None
        if ctx.needs_input_grad[2]: # bias
            grad2=gd0.sum((0,2,3))
        if ctx.needs_input_grad[1]: # weight
            x_padded = ctx.params['x_padded']
            grad1 = _conv2d_backward_w(xp, x_padded, gd0, stride, (0, 0), dilation, groups, xd1)[0]
        if ctx.needs_input_grad[0]:  # input
            grad0 = _conv2d_backward_x(xp, gd0, xd1, stride, padding, dilation, groups, x=xd0)
        return grad0, grad1, grad2


class TransposedConvolution(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs # input, weight, bias
        xd0, xd1 = xt0.data, xt1.data
        stride=params['stride']
        padding=params['padding']
        dilation=params['dilation']
        groups=params['groups']
        output_padding=params['output_padding']
        if xd0.ndim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv_transpose2d, '
                               f'but got input of size: {xd0.shape}')
        if xd1.shape[-4] != xd0.shape[-3]:
            raise RuntimeError(f'Given transposed=1, weight of size {xd1.shape}, '
                               f'expected input {xd0.shape} to have {xd1.shape[-4]} channels, '
                               f'but got {xd0.shape[-3]} channels instead')
        xp = ctx.xp
        yd0 = _conv2d_backward_x(xp, xd0, xd1, stride, padding, dilation, groups, output_padding=output_padding)
        if xt2 is not None:
            yd0 += xt2.data[:, None, None]
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1= ctx.saved_tensors
        stride = ctx.params['stride']
        padding = ctx.params['padding']
        dilation = ctx.params['dilation']
        groups = ctx.params['groups']
        grad_padded = None
        xp = ctx.xp
        grad0, grad1, grad2=None, None, None
        if ctx.needs_input_grad[2]: # bias
            grad2=gd0.sum((0,2,3))
        if ctx.needs_input_grad[1]: # weight
            grad1, grad_padded = _conv2d_backward_w(xp, gd0, xd0, stride, padding, dilation, groups, xd1)
        if ctx.needs_input_grad[0]:  # input
            grad0 = _conv2d(xp, gd0 if grad_padded is None else grad_padded, xd1, stride, (0, 0), dilation, groups)[0]
        return grad0, grad1, grad2


############ max pool2d

def _max_pool2d(xp, x, kernel_size, stride, padding, dilation, ceil_mode):
    low_dim_flag = False
    if x.ndim == 3:
        low_dim_flag = True
        x = x[None]
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
    idx1_m = idx1[ax0, ax1, ax2, ax3, idx2]
    pos = (ax0, ax1, ax2, ax3, idx2, idx1_m)
    if low_dim_flag:
        value = value[0]
    return value, pos


def _max_pool2d_backward(xp, x, grad, pos, kernel_size, stride, padding, dilation, ceil_mode):
    low_dim_flag = False
    if x.ndim == 3:
        low_dim_flag = True
        x = x[None]
    x_grad = xp.zeros_like(x)
    ax0, ax1, ax2, ax3, idx2, idx1_m = pos
    expanded, padded = _sliding_window_view(xp, x_grad, kernel_size, stride, dilation, padding, ceil_mode, val=xp.NINF)
    expanded[ax0, ax1, ax2, ax3, idx2, idx1_m] = xp.moveaxis(grad, -3, -1)
    hp, wp = padding
    h, w = x_grad.shape[-2:]
    x_grad = padded[..., hp:(hp + h), wp:(wp + w)]
    if low_dim_flag:
        x_grad = x_grad[0]
    return x_grad

class MaxPool2DWithIndices(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        kernel_size = params['kernel_size']
        stride = params['stride']
        padding = params['padding']
        dilation = params['dilation']
        ceil_mode = params['ceil_mode']
        return_indices = params['return_indices']
        if padding[0] * 2 > kernel_size[0] or padding[1] * 2 > kernel_size[1]:
            raise RuntimeError(f'pad should be smaller than or equal to half of kernel size, '
                               f'but got padW = {padding[1]}, padH = {padding[0]}, kW = {kernel_size[1]}, '
                               f'kH = {kernel_size[0]}')
        if xd0.ndim != 3 and xd0.ndim != 4:
            raise RuntimeError('non-empty 3D or 4D (batch mode) tensor expected for input')

        xp = ctx.xp
        yd0, pos = _max_pool2d(xp, xd0, kernel_size, stride, padding, dilation, ceil_mode)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0)
        ctx.params['pos'] = pos
        return (yt0, pos) if return_indices else yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, *_= grad_outputs
        xd0, = ctx.saved_tensors
        pos = ctx.params['pos']
        kernel_size = ctx.params['kernel_size']
        stride = ctx.params['stride']
        padding = ctx.params['padding']
        dilation = ctx.params['dilation']
        ceil_mode = ctx.params['ceil_mode']
        xp = ctx.xp
        grad0 = _max_pool2d_backward(xp, xd0, gd0, pos, kernel_size, stride, padding, dilation, ceil_mode)
        return grad0

class Droupout(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        p = params['p']
        xp = ctx.xp
        mask = xp.random.binomial(1, 1 - p, size=xd0.shape)
        if params['inplace']:
            inplace_precheck(xt0)
            xp.multiply(xd0, mask,out=xd0)
            if p!=1.:
                xp.divide(xd0, 1-p, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0=xp.multiply(xd0, mask, dtype=xd0.dtype)
            if p != 1.:
                yd0 = xp.divide(yd0, 1-p, out=yd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['mask']=mask
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        p = ctx.params['p']
        mask = ctx.params['mask']
        xp = ctx.xp
        grad0 = xp.multiply(gd0, mask, dtype=gd0.dtype)
        if p!=1.:
            grad0=xp.divide(grad0, 1-p, out=grad0)
        return grad0


def _verify_batch_size(size):
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    if size_prods == 1:
        raise ValueError(f'Expected more than 1 value per channel when training, got input size {size}')


class BatchNorm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs # input, weight, bias
        xd0 = xt0.data
        running_mean = ctx.params['running_mean']
        running_var = ctx.params['running_var']
        training = ctx.params['training']
        momentum = ctx.params['momentum']
        eps = ctx.params['eps']
        xp = ctx.xp
        axis = (0,) + tuple(range(2, xd0.ndim))
        shape = (xd0.shape[0],) + xd0.shape[2:]
        expanded_dims = (xd0.shape[1],)+(1,)*(xd0.ndim - 2)
        if training:
            _verify_batch_size(xd0.shape)
            batch_mean = xp.mean(xd0, axis=axis)
            batch_var = xp.var(xd0, axis=axis)
            if running_mean is not None and running_var is not None:
                # https://www.geeksforgeeks.org/python-multiply-numbers-list-3-different-ways/
                N = prod(shape)
                sample_var = batch_var * (N / (N - 1)) # calc N/(N-1) first, then multiply batch_var
                running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean
                running_var.data = (1 - momentum) * running_var.data + momentum * sample_var
        else:
            if running_mean is not None and running_var is not None:
                batch_mean = running_mean.data
                batch_var = running_var.data
            else:
                batch_mean = xp.mean(xd0, axis=axis)
                batch_var = xp.var(xd0, axis=axis)

        # don't use in-place batch_var += eps.
        # otherwise during test time (training=False and running_var exists),
        # the value of saved running_var will be increased everytime by eps.
        batch_var = batch_var + eps
        batch_sd = xp.sqrt(batch_var)

        # expand weight, bias running mean and var to (C,extra 1's), where number of extra 1's
        # is length of input dim-2
        batch_mean = batch_mean.reshape(expanded_dims)
        batch_var = batch_var.reshape(expanded_dims)
        batch_sd = batch_sd.reshape(expanded_dims)

        # affine
        yd0 = (xd0 - batch_mean) / batch_sd

        if xt1 is not None:
            xd1 = xt1.data
            yd0 *= xd1.reshape(expanded_dims)
        if xt2 is not None:
            xd2 = xt2.data
            yd0 += xd2.reshape(expanded_dims)
        yt0 = build_links(yd0, grad_fn=ctx)

        ctx.save_for_backward(xt0, xt1) # save input and weight for backward
        ctx.params={'axis': axis, 'shape':shape, 'mean': batch_mean, 'var':batch_var, 'sd':batch_sd}
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors # input, weight
        axis = ctx.params['axis']
        shape = ctx.params['shape']
        mu = ctx.params['mean']
        var = ctx.params['var']
        sd = ctx.params['sd']
        xp = ctx.xp
        grad0, grad1, grad2 = None, None, None
        if ctx.needs_input_grad[2]: # bias
            grad2=gd0.sum(axis)
        if ctx.needs_input_grad[1]:  # weight
            norm = (xd0 - mu) / sd
            grad1=(norm * gd0).sum(axis)
        if ctx.needs_input_grad[0]:  # input
            if xd1 is not None: # if weight is not None (i.e. when affine is True)
                gd0 = gd0*xd1.reshape(xd1.shape + (1,) * (xd0.ndim-2))
            N = prod(shape)
            a = N*gd0 - gd0.sum(axis=axis, keepdims=True)
            b = (xd0 - mu) / var * (gd0 * (xd0 - mu)).sum(axis=axis, keepdims=True)
            grad0 = xp.divide(a - b, N, dtype=gd0.dtype) / sd
        return grad0, grad1, grad2



class LayerNorm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs # input, weight, bias
        xd0 = xt0.data
        normalized_shape = tuple(ctx.params['normalized_shape'])
        eps = ctx.params['eps']
        if xt1 is not None and xt1.shape != normalized_shape:
            raise RuntimeError(
                f'Expected weight to be of same shape as normalized_shape, '
                f'but got weight of shape {xt1.shape} and normalized_shape = {normalized_shape}')
        if xt2 is not None and xt2.shape != normalized_shape:
            raise RuntimeError(f'Expected bias to be of same shape as normalized_shape, '
                               f'but got bias of shape {xt2.shape} and normalized_shape = {normalized_shape}')
        if xd0.shape[-len(normalized_shape):] != normalized_shape:
            raise RuntimeError(f'Given normalized_shape={normalized_shape}, '
                               f'expected input with shape (*, {", ".join([str(_) for _ in normalized_shape])}), '
                               f'but got input of size{xd0.shape}')
        xp = ctx.xp
        axis = tuple(-i for i in range(1, 1 + len(normalized_shape)))
        shape = xd0.shape[-len(normalized_shape):]
        layer_mean = xp.mean(xd0, axis=axis, keepdims=True)
        layer_var = xp.var(xd0, axis=axis, keepdims=True) + eps
        layer_sd = xp.sqrt(layer_var)
        # elementwise affine
        yd0 = (xd0 - layer_mean) / layer_sd
        if xt1 is not None:
            yd0 *= xt1.data
        if xt2 is not None:
            yd0 += xt2.data
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1) # save input and weight for backward
        ctx.params = {'axis': axis, 'shape': shape, 'mean': layer_mean, 'var': layer_var, 'sd': layer_sd}
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors # input, weight
        axis = ctx.params['axis']
        shape = ctx.params['shape']
        mu = ctx.params['mean']
        var = ctx.params['var']
        sd = ctx.params['sd']
        xp = ctx.xp
        grad0, grad1, grad2 = None, None, None
        sum_axis=tuple(range(gd0.ndim - len(axis)))
        if ctx.needs_input_grad[2]: # bias
            grad2=gd0.sum(sum_axis)
        if ctx.needs_input_grad[1]:  # weight
            norm = (xd0 - mu) / sd
            grad1=(norm * gd0).sum(sum_axis)
        if ctx.needs_input_grad[0]:  # input
            if xd1 is not None: # if weight is not None (i.e. when affine is True)
                gd0 = gd0*xd1
            N = prod(shape)
            a = N*gd0 - gd0.sum(axis=axis, keepdims=True)
            b = (xd0 - mu) / var * (gd0 * (xd0 - mu)).sum(axis=axis, keepdims=True)
            grad0 = xp.divide(a - b, N, dtype=gd0.dtype) / sd
        return grad0, grad1, grad2

class Embedding(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0,=inputs # weight
        xd0=xt0.data
        input=ctx.params['input']
        padding_idx=ctx.params['padding_idx']
        max_norm = ctx.params['max_norm']
        norm_type = ctx.params['norm_type']
        sparse = ctx.params['sparse']
        if not isinstance(input, Tensor):
            raise TypeError(f'embedding(): input must be Tensor, not {input.__class__.__name__}')
        if not sparse_is_loaded and sparse:
            raise RuntimeError('Numpy do not support sparse, set sparse=False, or install cupy/scipy')
        # change padding index to positive
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < xd0.shape[0], "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert padding_idx >= -xd0.shape[0], "Padding_idx must be within num_embeddings"
                padding_idx = xd0.shape[0] + padding_idx
        else:
            padding_idx = -1
        xp = ctx.xp
        # renorm if max_norm is not None
        indices, counts = None, None
        if max_norm is not None:
            indices, counts = xp.unique(xp.ascontiguousarray(input.data), return_counts=True)
            norm = xp.linalg.norm(xd0[indices], ord=norm_type, axis=-1, keepdims=True)
            rescale = (norm > max_norm)[:, 0]
            selection = xp.zeros(xd0.shape[0], dtype=bool)
            selection[indices] = rescale
            xd0[selection] *= max_norm / (norm[rescale] + 1e-7)
        yt0 = build_links(xd0[input.data], grad_fn=ctx)
        ctx.save_for_backward(xt0)
        ctx.params['counts'] = counts
        ctx.params['indices'] = indices
        ctx.params['padding_idx'] = padding_idx
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        be careful to not back-propagate further when sparse is True, because grad0 is then sparse matrix,
        not xparray.
        """
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        input = ctx.params['input'].data
        padding_idx = ctx.params['padding_idx']  # can be None
        scale_grad_by_freq = ctx.params['scale_grad_by_freq']  # bool
        sparse = ctx.params['sparse']  # bool
        indices = ctx.params['indices']
        counts = ctx.params['counts']
        xp = ctx.xp

        inpt_flatten = input.flatten()
        grad_reshaped = gd0.reshape(-1, gd0.shape[-1])  # to 2d matrix
        if sparse:
            csr_matrix = cp_sparse.csr_matrix if xp is cp else sci_sparse.csr_matrix
            # argsort input and grad
            p = inpt_flatten.argsort()
            grad_reshaped = grad_reshaped[p]
            inpt_flatten = inpt_flatten[p]
            # input frequency count
            bin_count = xp.bincount(inpt_flatten + 1, minlength=xd0.shape[-2] + 1)
            if scale_grad_by_freq:
                grad_reshaped /= bin_count[inpt_flatten + 1][..., None]
            # construct sparse grad
            data = grad_reshaped.flatten()
            ind = xp.tile(xp.arange(xd0.shape[-1]), len(inpt_flatten))
            indptr = (xd0.shape[-1] * bin_count).cumsum()
            value = csr_matrix((data, ind, indptr), shape=xd0.shape)
            # pytorch doesn't sum duplicates, no need for value.sum_duplicates()
            # padding index
            if padding_idx >= 0:
                value.data[value.indptr[padding_idx]:value.indptr[padding_idx + 1]] = 0
        else:
            scatter_add = cpx.scatter_add if xp is cp else np.add.at
            value = xp.zeros_like(xd0)
            scatter_add(value, inpt_flatten, grad_reshaped)
            if padding_idx >= 0:
                value[padding_idx] = 0
            if scale_grad_by_freq:
                if indices is None:
                    indices, counts = xp.unique(input.data, return_counts=True)
                value[indices] /= counts[:, None]
        grad0=value
        return grad0

