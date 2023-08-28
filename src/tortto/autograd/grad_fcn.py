from .function import *
from .helper import *
from tortto import np, cp, cparray

"""
Auto-generated from grad_fcn_generator.py
Any changes to this file will NOT be kept during next import
Instead, make changes to grad_fcn_config.yaml to take effect
"""


class Sqrt(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.sqrt(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.sqrt(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        y0, = ctx.saved_tensors
        grad0 = g0 / (y0 * 2)
        return grad0


class Exp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.exp(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        y0, = ctx.saved_tensors
        grad0 = g0 * y0
        return grad0


class Tan(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tan(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.tan(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        y0, = ctx.saved_tensors
        grad0 = g0 * (1 + y0 * y0)
        return grad0


class Tanh(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tanh(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.tanh(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        y0, = ctx.saved_tensors
        grad0 = g0 * (1 - y0 * y0)
        return grad0


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(-xp.logaddexp(0, -x0, out=x0), out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.exp(-xp.logaddexp(0, -x0))
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        y0, = ctx.saved_tensors
        grad0 = g0 * y0 * (1 - y0)
        return grad0


class Sign(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.sign(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.sign(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        xp = ctx.xp
        grad0 = xp.zeros_like(g0)
        return grad0


class Neg(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.negative(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.negative(x0)
            yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        grad0 = -g0
        return grad0


class Add(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.add(x0, x1, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.add(x0, x1)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = (x0.shape, x1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_shape, x1_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0, x0_shape)
        if req_grad[1]:
            grad1 = reverse_broadcast(g0, x1_shape)
        return grad0, grad1


class Sub(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            xp.subtract(x0, x1, out=x0)
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = xp.subtract(x0, x1)
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = (x0.shape, x1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_shape, x1_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0, x0_shape)
        if req_grad[1]:
            grad1 = -reverse_broadcast(g0, x1_shape)
        return grad0, grad1


class Sin(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.sin(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            y0 = xp.sin(x0)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        if x0 is None:
            x0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = g0 * xp.cos(x0)
        return grad0


class Cos(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.cos(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            y0 = xp.cos(x0)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        if x0 is None:
            x0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = g0 * -xp.sin(x0)
        return grad0


class Log(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.log(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            y0 = xp.log(x0)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        if x0 is None:
            x0 = ctx.params['copy']
        grad0 = g0 / x0
        return grad0


class Abs(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.abs(x0, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            y0 = xp.abs(x0)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        if x0 is None:
            x0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = g0 * xp.sign(x0)
        return grad0


class Pow(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.power(x0, x1, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1 if req_grad[0] else None, yt0)
        else:
            y0 = xp.power(x0, x1)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1 if req_grad[0] else None, yt0)
        ctx.params['property'] = x1.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1, y0 = ctx.saved_tensors
        x1_shape = ctx.params['property']
        if x0 is None:
            x0 = ctx.params['copy']
        xp = ctx.xp
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0 * x1 * y0 / x0, x0.shape)
        if req_grad[1]:
            grad1 = reverse_broadcast(g0 * xp.log(x0) * y0, x1_shape)
        return grad0, grad1


class Mul(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if req_grad[1]:
                ctx.params['copy'] = x0.copy()
            xp.multiply(x0, x1, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1 if req_grad[0] else None)
        else:
            y0 = xp.multiply(x0, x1)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0 if req_grad[1] else None, xt1 if req_grad[0] else None)
        ctx.params['property'] = (x0.shape, x1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        x0_shape, x1_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0 * x1, x0_shape)
        if req_grad[1]:
            if x0 is None:
                x0 = ctx.params['copy']
            grad1 = reverse_broadcast(g0 * x0, x1_shape)
        return grad0, grad1


class Div(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if req_grad[1]:
                ctx.params['copy'] = x0.copy()
            xp.divide(x0, x1, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            y0 = xp.divide(x0, x1)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0 if req_grad[1] else None, xt1)
        ctx.params['property'] = x0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        x0_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0 / x1, x0_shape)
        if req_grad[1]:
            if x0 is None:
                x0 = ctx.params['copy']
            grad1 = reverse_broadcast(-g0 * x0 / (x1 * x1), x1.shape)
        return grad0, grad1


class Clamp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        max = params['max']
        min = params['min']
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = x0.copy()
            xp.clip(x0, a_min=min, a_max=max, out=x0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            y0 = xp.clip(x0, a_min=min, a_max=max)
            yt0 = build_links(y0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        max = ctx.params['max']
        min = ctx.params['min']
        if x0 is None:
            x0 = ctx.params['copy']
        grad0 = g0
        if min is not None:
            g0[x0 < min] = 0
        if max is not None:
            g0[x0 > max] = 0
        return grad0


class Max0(Function):
    # optimize it?
    # https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max"
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        y0 = xp.max(x0, axis=dim, keepdims=keepdim)
        yt0 = build_links(y0, grad_fn=ctx)
        argmax = xp.argmax(x0, axis=dim, keepdims=keepdim)
        y1 = argmax
        yt1 = tt.tensor(y1, copy=False)
        ctx.params['arrays'] = argmax
        ctx.params['property'] = x0.shape
        return yt0, yt1

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, _ = grad_outputs
        argmax = ctx.params['arrays']
        x0_shape = ctx.params['property']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        idx = xp.ogrid[[slice(ax) for ax in argmax.shape]]
        if keepdim:
            idx[dim] = argmax
        else:
            idx.insert(dim, argmax)
        grad0 = xp.zeros(x0_shape, dtype=g0.dtype)
        grad0[tuple(idx)] = g0
        return grad0


class Min0(Function):
    # optimize it?
    # https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max"
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        y0 = xp.min(x0, axis=dim, keepdims=keepdim)
        yt0 = build_links(y0, grad_fn=ctx)
        argmin = xp.argmin(x0, axis=dim, keepdims=keepdim)
        y1 = argmin
        yt1 = tt.tensor(y1, copy=False)
        ctx.params['arrays'] = argmin
        ctx.params['property'] = x0.shape
        return yt0, yt1

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, _ = grad_outputs
        argmin = ctx.params['arrays']
        x0_shape = ctx.params['property']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        idx = xp.ogrid[[slice(ax) for ax in argmin.shape]]
        if keepdim:
            idx[dim] = argmin
        else:
            idx.insert(dim, argmin)
        grad0 = xp.zeros(x0_shape, dtype=g0.dtype)
        grad0[tuple(idx)] = g0
        return grad0


class Max1(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        y0 = xp.max(x0)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0, yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, y0 = ctx.saved_tensors
        xp = ctx.xp
        grad0 = xp.zeros_like(x0)
        grad0[x0 == y0] = g0
        return grad0


class Min1(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        xp = ctx.xp
        y0 = xp.min(x0)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0, yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, y0 = ctx.saved_tensors
        xp = ctx.xp
        grad0 = xp.zeros_like(x0)
        grad0[x0 == y0] = g0
        return grad0


class Maximum(Function):
    # optimize it?
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        xp = ctx.xp
        y0 = xp.maximum(x0, x1)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        xp = ctx.xp
        req_grad = ctx.needs_input_grad
        maximum = xp.maximum(x0, x1)
        x0_equal_max_ind = maximum == x0
        x1_equal_max_ind = maximum == x1
        both_equal_max_ind = x0_equal_max_ind & x1_equal_max_ind
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = g0.copy() if req_grad[1] else g0
            grad0[~x0_equal_max_ind] = 0
            grad0[both_equal_max_ind] /= 2
            grad0 = reverse_broadcast(grad0, x0.shape)
        if req_grad[1]:
            grad1 = g0
            grad1[~x1_equal_max_ind] = 0
            grad1[both_equal_max_ind] /= 2
            grad1 = reverse_broadcast(grad1, x1.shape)
        return grad0, grad1


class Minimum(Function):
    # optimize it?
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        xp = ctx.xp
        y0 = xp.minimum(x0, x1)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        xp = ctx.xp
        req_grad = ctx.needs_input_grad
        minimum = xp.minimum(x0, x1)
        x0_equal_min_ind = minimum == x0
        x1_equal_min_ind = minimum == x1
        both_equal_min_ind = x0_equal_min_ind & x1_equal_min_ind
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = g0.copy() if req_grad[1] else g0
            grad0[~x0_equal_min_ind] = 0
            grad0[both_equal_min_ind] /= 2
            grad0 = reverse_broadcast(grad0, x0.shape)
        if req_grad[1]:
            grad1 = g0
            grad1[~x1_equal_min_ind] = 0
            grad1[both_equal_min_ind] /= 2
            grad1 = reverse_broadcast(grad1, x1.shape)
        return grad0, grad1


class View(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        shape = params['shape']
        y0 = x0.reshape(shape)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = x0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_shape = ctx.params['property']
        grad0 = g0.reshape(x0_shape)
        return grad0


class Slice(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        key = params['key']
        y0 = x0[key]
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = x0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_shape = ctx.params['property']
        key = ctx.params['key']
        xp = ctx.xp
        grad0 = xp.zeros(x0_shape, dtype=g0.dtype)
        grad0[key] = g0
        return grad0


class Permute(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dims = params['dims']
        xp = ctx.xp
        y0 = xp.transpose(x0, dims)
        yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        dims = ctx.params['dims']
        xp = ctx.xp
        grad0 = xp.transpose(g0, axes=np.argsort(dims))
        return grad0


class Transpose(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim0 = params['dim0']
        dim1 = params['dim1']
        xp = ctx.xp
        y0 = xp.swapaxes(x0, dim0, dim1)
        yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        dim0 = ctx.params['dim0']
        dim1 = ctx.params['dim1']
        xp = ctx.xp
        grad0 = xp.swapaxes(g0, dim0, dim1)
        return grad0


class Squeeze(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        if dim.__class__ is int:
            dim = (dim,)
        if dim is None:
            dim = tuple(range(x0.ndim))
        squeeze_dims = tuple(i for i in dim if x0.shape[i] == 1)
        if len(squeeze_dims) == 0:
            y0 = x0
        else:
            y0 = xp.squeeze(x0, squeeze_dims)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['arrays'] = squeeze_dims
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        squeeze_dims = ctx.params['arrays']
        xp = ctx.xp
        grad0 = xp.expand_dims(g0, squeeze_dims)
        return grad0


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        y0 = xp.expand_dims(x0, dim)
        yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        dim = ctx.params['dim']
        xp = ctx.xp
        grad0 = xp.squeeze(g0, dim)
        return grad0


class Repeat(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        sizes = params['sizes']
        xp = ctx.xp
        y0 = xp.tile(x0, sizes)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = (x0.ndim, x0.shape, y0.strides)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_ndim, x0_shape, y0_strides = ctx.params['property']
        sizes = ctx.params['sizes']
        xp = ctx.xp
        leading_dims = tuple(range(len(sizes)))
        target_shape = sizes + x0_shape
        target_strides = y0_strides[:-x0_ndim] + tuple(
            x0_shape[i] * y0_strides[i - x0_ndim] for i in range(x0_ndim)) + y0_strides[-x0_ndim:]
        grad0 = xp.lib.stride_tricks.as_strided(g0, shape=target_shape, strides=target_strides).sum(leading_dims)
        return grad0


class ToCopy(Function):
    # no-op if same device. use `return xt0` to represent the input tensor
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        target_device = params['target_device']
        xp = ctx.xp
        if xp is cp:
            if target_device == 'cuda':
                return xt0
            else:
                y0 = x0.get()
        else:
            if target_device == 'cpu':
                return xt0
            else:
                y0 = cparray(x0)
        yt0 = build_links(y0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        target_device = ctx.params['target_device']
        if target_device == 'cpu':
            grad0 = cp.array(g0)
        else:
            grad0 = g0.get()
        return grad0


class Cat(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        dim = params['dim']
        xp = ctx.xp
        xn = []
        indices = []
        for xt in inputs:
            xn.append(xt.data)
            indices.append(xt.shape[dim])
        indices = np.cumsum(indices)
        y0 = xp.concatenate(xn, dim)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['arrays'] = indices
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        indices = ctx.params['arrays']
        dim = ctx.params['dim']
        xp = ctx.xp
        grad0 = xp.split(g0, indices_or_sections=indices[:-1], axis=dim)
        return grad0


class Mm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        if x0.ndim != 2:
            raise RuntimeError('self must be a matrix')
        if x1.ndim != 2:
            raise RuntimeError('mat2 must be a matrix')
        y0 = x0 @ x1
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0 if req_grad[1] else None, xt1 if req_grad[0] else None)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = g0 @ x1.T
        if req_grad[1]:
            grad1 = x0.T @ g0
        return grad0, grad1


class Mv(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        if x0.ndim != 2:
            raise RuntimeError('input must be a matrix')
        if x1.ndim != 1:
            raise RuntimeError('vec must be a vector')
        y0 = x0 @ x1
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0 if req_grad[1] else None, xt1 if req_grad[0] else None)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = g0[:, None] @ x1[None]
        if req_grad[1]:
            grad1 = x0.T @ g0
        return grad0, grad1


class Bmm(Function):
    # This is different from torch bmm. It deals with all cases of matmul except when matrices are 1D/2D
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        req_grad = ctx.needs_input_grad
        y0 = x0 @ x1
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0 if req_grad[1] else None, xt1 if req_grad[0] else None)
        ctx.params['property'] = (x0.shape, x1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, x1 = ctx.saved_tensors
        x0_shape, x1_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0 @ x1.swapaxes(-1, -2), x0_shape)
        if req_grad[1]:
            grad1 = reverse_broadcast(x0.swapaxes(-1, -2) @ g0, x1_shape)
        return grad0, grad1


class Addmm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs
        x0, x1, x2 = xt0.data, xt1.data, xt2.data
        alpha = params['alpha']
        beta = params['beta']
        req_grad = ctx.needs_input_grad
        if x1.ndim != 2:
            raise RuntimeError(f'mat1 must be a matrix, got {x1.ndim}-D tensor')
        if x2.ndim != 2:
            raise RuntimeError(f'mat2 must be a matrix, got {x2.ndim}-D tensor')
        y0 = alpha * (x1 @ x2) if beta == 0 else beta * x0 + alpha * (x1 @ x2)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt1 if req_grad[2] else None, xt2 if req_grad[1] else None)
        ctx.params['property'] = x0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x1, x2 = ctx.saved_tensors
        x0_shape = ctx.params['property']
        req_grad = ctx.needs_input_grad
        grad0, grad1, grad2 = None, None, None
        if req_grad[0]:
            grad0 = reverse_broadcast(g0, x0_shape)
        if req_grad[1]:
            grad1 = g0 @ x2.T
        if req_grad[2]:
            grad2 = x1.T @ g0
        return grad0, grad1, grad2


class Sum(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        y0 = xp.sum(x0, axis=dim, keepdims=keepdim)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = (x0.ndim, x0.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_ndim, x0_shape = ctx.params['property']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        if dim is None:
            grad0 = xp.lib.stride_tricks.as_strided(g0, shape=x0_shape, strides=(0,) * x0_ndim)
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                g0 = xp.expand_dims(g0, dim)
            strides = list(g0.strides)
            for i in dim:
                strides[i] = 0  # repeat along axis in x.shape
            grad0 = xp.lib.stride_tricks.as_strided(g0, shape=x0_shape, strides=strides)
        return grad0


class Mean(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        xp = ctx.xp
        y0 = xp.mean(x0, axis=dim, keepdims=keepdim)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['property'] = (x0.ndim, x0.shape, x0.size)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0_ndim, x0_shape, x0_size = ctx.params['property']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        if dim is None:
            grad0 = xp.lib.stride_tricks.as_strided(
                xp.divide(g0, x0_size, dtype=g0.dtype), shape=x0_shape, strides=(0,) * x0_ndim
            )
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                g0 = xp.expand_dims(g0, dim)
            N = 1
            strides = list(g0.strides)
            for i in dim:
                N *= x0_shape[i]
                strides[i] = 0  # repeat along axis in x.shape
            grad0 = xp.lib.stride_tricks.as_strided(xp.divide(g0, N, dtype=g0.dtype), shape=x0_shape, strides=strides)
        return grad0


class Var(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        x0 = xt0.data
        dim = params['dim']
        keepdim = params['keepdim']
        unbiased = params['unbiased']
        xp = ctx.xp
        y0 = xp.var(x0, axis=dim, ddof=unbiased, keepdims=keepdim)
        yt0 = build_links(y0, grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        x0, = ctx.saved_tensors
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        unbiased = ctx.params['unbiased']
        xp = ctx.xp
        if not keepdim:
            g0 = xp.expand_dims(g0, dim)
        mean = xp.mean(x0, axis=dim, keepdims=True)
        if dim is None:
            N = x0.size
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            N = 1
            for i in dim:
                N *= x0.shape[i]
        grad0 = 2 * g0 * xp.divide(x0 - mean, N - unbiased, dtype=g0.dtype)
        return grad0


class CopySlices(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        key = params['key']
        flag = None
        if x0.__class__ is cparray and x1.__class__ is not cparray:
            x1 = cp.array(x1)
            flag = True
        elif x0.__class__ is not cparray and x1.__class__ is cparray:
            x1 = x1.get()
            flag = False
        inplace_precheck(xt0)
        x0[key] = x1
        yt0 = inplace_update(xt0, ctx)
        ctx.params['arrays'] = flag if ctx.needs_input_grad[1] else None
        ctx.params['property'] = x1.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        flag = ctx.params['arrays']
        x1_shape = ctx.params['property']
        key = ctx.params['key']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[1]:
            # grad for value. Do this first because gd0 will be changed inplace next
            grad1 = reverse_broadcast(g0[key], x1_shape)
            if flag is True:
                grad1 = grad1.get()
            elif flag is False:
                grad1 = cp.array(grad1)
        if req_grad[0]:
            grad0 = g0  # grad for input
            grad0[key] = 0
        return grad0, grad1


class Copy(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        flag = None
        if x0.__class__ is cparray and x1.__class__ is not cparray:
            x1 = cp.array(x1)
            flag = True
        elif x0.__class__ is not cparray and x1.__class__ is cparray:
            x1 = x1.get()
            flag = False
        inplace_precheck(xt0)
        x0[...] = x1
        yt0 = inplace_update(xt0, ctx)
        ctx.params['arrays'] = flag if ctx.needs_input_grad[1] else None
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        flag = ctx.params['arrays']
        req_grad = ctx.needs_input_grad
        grad0, grad1 = None, None
        if req_grad[1]:
            # grad for value. Do this first because gd0 will be changed inplace next
            grad1 = g0
            if flag is True:
                grad1 = grad1.get()
            elif flag is False:
                grad1 = cp.array(grad1)
        if req_grad[0]:
            grad0 = g0  # grad for input
            grad0[...] = 0
        return grad0, grad1


class MaskedFill(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        x0, x1 = xt0.data, xt1.data
        mask = params['mask']
        if x1.ndim > 0:
            raise RuntimeError(f"masked_fill only supports a 0-dimensional value tensor, "
                               f"but got tensor with {x1.ndim} dimension(s).")
        if mask.dtype.type is not np.bool_:
            raise RuntimeError(f"dtype of mask must be bool. Pass dtype=bool when constructing mask")
        flag = False
        if x0.__class__ is cparray and x1.__class__ is not cparray:  # xd1 is a scaler, no need to convert it to cparray
            flag = True
        elif x0.__class__ is not cparray and x1.__class__ is cparray:
            raise RuntimeError(f"masked_fill: Expected inputs to be on same device")
        key = (slice(None),) * (x0.ndim - mask.ndim) + (mask.data,)
        if params['inplace']:
            inplace_precheck(xt0)
            x0[key] = x1
            yt0 = inplace_update(xt0, ctx)
        else:
            y0 = x0.copy()
            y0[key] = x1
            yt0 = build_links(y0, grad_fn=ctx)
        ctx.params['arrays'] = flag if ctx.needs_input_grad[1] else None
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        g0, = grad_outputs
        flag = ctx.params['arrays']
        mask = ctx.params['mask']
        req_grad = ctx.needs_input_grad
        leading = (slice(None),) * (g0.ndim - mask.ndim)
        key = leading + (mask.data,)
        grad0, grad1 = None, None
        if req_grad[1]:
            grad1 = g0[key].sum()
            if flag:
                grad1 = grad1.get()
        if req_grad[0]:
            grad0 = g0
            grad0[key] = 0
        return grad0, grad1
