from .function import *
from .helper import *
from tortto import cp, cparray, cupy_is_loaded

"""
Auto-generated from grad_ufunc_generator.py
Any changes to this file will NOT be kept during next import
Instead, make changes to grad_ufunc_config.yaml to take effect
"""


class Sqrt(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.sqrt(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.sqrt(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 / (yd0 * 2)
        return grad0


class Exp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.exp(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * yd0
        return grad0


class Tan(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tan(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.tan(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * (1 + yd0 * yd0)
        return grad0


class Tanh(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tanh(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.tanh(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * (1 - yd0 * yd0)
        return grad0


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(-xp.logaddexp(0, -xd0, out=xd0), out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.exp(-xp.logaddexp(0, -xd0))
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * yd0 * (1 - yd0)
        return grad0


class Sign(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        yd0 = xp.sign(xd0)
        yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xp = ctx.xp
        grad0 = xp.zeros_like(gd0)
        return grad0


class Neg(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.negative(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.negative(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0 = -gd0
        return grad0


class Add(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.add(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.add(xd0, xd1)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = reverse_broadcast(gd0, xd1_shape)
        return grad0, grad1


class Sub(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.subtract(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
        else:
            yd0 = xp.subtract(xd0, xd1)
            yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = -reverse_broadcast(gd0, xd1_shape)
        return grad0, grad1


class Sin(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.sin(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yd0 = xp.sin(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = gd0 * xp.cos(xd0)
        return grad0


class Cos(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.cos(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yd0 = xp.cos(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = gd0 * -xp.sin(xd0)
        return grad0


class Log(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.log(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yd0 = xp.log(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        grad0 = gd0 / xd0
        return grad0


class Abs(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.abs(xd0, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yd0 = xp.abs(xd0)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = gd0 * xp.sign(xd0)
        return grad0


class Pow(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.power(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1, yt0)
        else:
            yd0 = xp.power(xd0, xd1)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1, yt0)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1, yd0 = ctx.saved_tensors
        xd0_shape, xd1_shape = ctx.params['shape']
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 * xd1 * yd0 / xd0, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = reverse_broadcast(gd0 * xp.log(xd0) * yd0, xd1_shape)
        return grad0, grad1


class Mul(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.multiply(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yd0 = xp.multiply(xd0, xd1)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        xd0_shape, xd1_shape = ctx.params['shape']
        if xd0 is None:
            xd0 = ctx.params['copy']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 * xd1, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = reverse_broadcast(gd0 * xd0, xd1_shape)
        return grad0, grad1


class Div(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.divide(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yd0 = xp.divide(xd0, xd1)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        xd0_shape, xd1_shape = ctx.params['shape']
        if xd0 is None:
            xd0 = ctx.params['copy']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 / xd1, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = reverse_broadcast(-gd0 * xd0 / (xd1 * xd1), xd1_shape)
        return grad0, grad1


class Clamp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        min = params['min']
        max = params['max']
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.clip(xd0, a_min=min, a_max=max, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yd0 = xp.clip(xd0, a_min=min, a_max=max)
            yt0 = build_links(yd0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        min = ctx.params['min']
        max = ctx.params['max']
        if xd0 is None:
            xd0 = ctx.params['copy']
        grad0 = gd0
        if min is not None:
            gd0[xd0 < min] = 0
        if max is not None:
            gd0[xd0 > max] = 0
        return grad0


class Max0(Function):
    # optimize it?
    # https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        yd0 = xp.max(xd0, axis=dim, keepdims=keepdim)
        yt0 = build_links(yd0, grad_fn=ctx)
        argmax = xp.argmax(xd0, axis=dim, keepdims=keepdim)
        yd1 = argmax
        yt1 = tt.tensor(yd1, copy=False)
        ctx.params['argmax'] = argmax
        ctx.params['shape'] = xd0.shape
        return yt0, yt1

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, _ = grad_outputs
        argmax = ctx.params['argmax']
        xd0_shape = ctx.params['shape']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        idx = xp.ogrid[[slice(ax) for ax in argmax.shape]]
        if keepdim:
            idx[dim] = argmax
        else:
            idx.insert(dim, argmax)
        grad0 = xp.zeros(xd0_shape, dtype=gd0.dtype)
        grad0[tuple(idx)] = gd0
        return grad0


class Max1(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        yd0 = xp.max(xd0)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, yd0 = ctx.saved_tensors
        xp = ctx.xp
        grad0 = xp.zeros_like(xd0)
        grad0[xd0 == yd0] = gd0
        return grad0


class Maximum(Function):
    # optimize it?
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        yd0 = xp.maximum(xd0, xd1)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        xd0_shape, xd1_shape = ctx.params['shape']
        xp = ctx.xp
        maximum = xp.maximum(xd0, xd1)
        xd0_equal_max_ind = maximum == xd0
        xd1_equal_max_ind = maximum == xd1
        both_equal_max_ind = xd0_equal_max_ind & xd1_equal_max_ind
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = gd0.copy() if ctx.needs_input_grad[1] else gd0
            grad0[~xd0_equal_max_ind] = 0
            grad0[both_equal_max_ind] /= 2
            grad0 = reverse_broadcast(grad0, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = gd0
            grad1[~xd1_equal_max_ind] = 0
            grad1[both_equal_max_ind] /= 2
            grad1 = reverse_broadcast(grad1, xd1_shape)
        return grad0, grad1


class Min0(Function):
    # optimize it?
    # https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        yd0 = xp.min(xd0, axis=dim, keepdims=keepdim)
        yt0 = build_links(yd0, grad_fn=ctx)
        argmin = xp.argmin(xd0, axis=dim, keepdims=keepdim)
        yd1 = argmin
        yt1 = tt.tensor(yd1, copy=False)
        ctx.params['argmin'] = argmin
        ctx.params['shape'] = xd0.shape
        return yt0, yt1

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, _ = grad_outputs
        argmin = ctx.params['argmin']
        xd0_shape = ctx.params['shape']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        idx = xp.ogrid[[slice(ax) for ax in argmin.shape]]
        if keepdim:
            idx[dim] = argmin
        else:
            idx.insert(dim, argmin)
        grad0 = xp.zeros(xd0_shape, dtype=gd0.dtype)
        grad0[tuple(idx)] = gd0
        return grad0


class Min1(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        yd0 = xp.min(xd0)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, yd0 = ctx.saved_tensors
        xp = ctx.xp
        grad0 = xp.zeros_like(xd0)
        grad0[xd0 == yd0] = gd0
        return grad0


class Minimum(Function):
    # optimize it?
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        yd0 = xp.minimum(xd0, xd1)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        xd0_shape, xd1_shape = ctx.params['shape']
        xp = ctx.xp
        minimum = xp.minimum(xd0, xd1)
        xd0_equal_min_ind = minimum == xd0
        xd1_equal_min_ind = minimum == xd1
        both_equal_min_ind = xd0_equal_min_ind & xd1_equal_min_ind
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = gd0.copy() if ctx.needs_input_grad[1] else gd0
            grad0[~xd0_equal_min_ind] = 0
            grad0[both_equal_min_ind] /= 2
            grad0 = reverse_broadcast(grad0, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = gd0
            grad1[~xd1_equal_min_ind] = 0
            grad1[both_equal_min_ind] /= 2
            grad1 = reverse_broadcast(grad1, xd1_shape)
        return grad0, grad1


class View(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        shape = params['shape']
        yd0 = xd0.reshape(shape)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['shape'] = xd0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        grad0 = gd0.reshape(xd0_shape)
        return grad0


class Slice(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        key = params['key']
        yd0 = xd0[key]
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['shape'] = xd0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        key = ctx.params['key']
        xp = ctx.xp
        grad0 = xp.zeros(xd0_shape, dtype=gd0.dtype)
        grad0[key] = gd0
        return grad0


class Permute(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dims = params['dims']
        xp = ctx.xp
        yd0 = xp.transpose(xd0, dims)
        yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dims = ctx.params['dims']
        xp = ctx.xp
        grad0 = xp.transpose(gd0, axes=tt.np.argsort(dims))
        return grad0


class Transpose(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim0 = params['dim0']
        dim1 = params['dim1']
        xp = ctx.xp
        yd0 = xp.swapaxes(xd0, dim0, dim1)
        yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim0 = ctx.params['dim0']
        dim1 = ctx.params['dim1']
        xp = ctx.xp
        grad0 = xp.swapaxes(gd0, dim0, dim1)
        return grad0


class Squeeze(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        if dim.__class__ is int:
            dim = (dim,)
        if dim is None:
            dim = tuple(range(xd0.ndim))
        squeeze_dims = tuple(i for i in dim if xd0.shape[i] == 1)
        if len(squeeze_dims) == 0:
            yd0 = xd0
        else:
            yd0 = xp.squeeze(xd0, squeeze_dims)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['squeeze_dims'] = squeeze_dims
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        squeeze_dims = ctx.params['squeeze_dims']
        xp = ctx.xp
        grad0 = xp.expand_dims(gd0, squeeze_dims)
        return grad0


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        yd0 = xp.expand_dims(xd0, dim)
        yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim = ctx.params['dim']
        xp = ctx.xp
        grad0 = xp.squeeze(gd0, dim)
        return grad0


class Repeat(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        sizes = params['sizes']
        xp = ctx.xp
        yd0 = xp.tile(xd0, sizes)
        yd0_strides = yd0.strides
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.params['yd0_strides'] = yd0_strides
        ctx.params['shape'] = xd0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0_strides = ctx.params['yd0_strides']
        xd0_shape = ctx.params['shape']
        sizes = ctx.params['sizes']
        xp = ctx.xp
        xd0_ndim = len(xd0_shape)
        leading_dims = tuple(range(len(sizes)))
        target_shape = sizes + xd0_shape
        target_strides = yd0_strides[:-xd0_ndim] + tuple(
            xd0_shape[i] * yd0_strides[i - xd0_ndim] for i in range(xd0_ndim)) + yd0_strides[-xd0_ndim:]
        grad0 = xp.lib.stride_tricks.as_strided(gd0, shape=target_shape, strides=target_strides).sum(leading_dims)
        return grad0


class ToCopy(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        target_device = params['target_device']
        xp = ctx.xp
        if xp is cp:
            if target_device == 'cuda':
                return xt0
            else:
                yd0 = xd0.get()
        else:
            if target_device == 'cpu':
                return xt0
            else:
                #if not cupy_is_loaded:
                 #   raise RuntimeError('cupy not installed, can\'t use cuda')
                yd0 = cparray(xd0)
        yt0 = build_links(yd0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        target_device = ctx.params['target_device']
        if target_device == 'cpu':
            grad0 = cp.array(gd0)
        else:
            grad0 = gd0.get()
        return grad0
