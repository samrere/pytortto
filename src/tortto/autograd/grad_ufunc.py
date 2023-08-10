from .function import *
from .helper import *


"""
'x' is input
'y' is output
'g' is gradient

't' is for tensor
'd' is for data (xparray)

Use special formatting if the function allows inplace, but not all tensors in saved_tensors are used during backward.
Example: in Div, saved tensor xd0 (numerator) is not used during backward for numerator.
Therefore, if the denominator doesn't require grad, xd0 can be changed inplace and backward still works.
Same goes for Mul.

import torch
x=torch.tensor([1,2,3.], requires_grad=True)+0
y=torch.tensor([4,5,6.], requires_grad=False)*1
z=x/y
x+=1
z.backward(torch.tensor([1,1,1]))
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
            yt0 = build_links(xp.sqrt(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.exp(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.tan(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.tanh(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.exp(-xp.logaddexp(0, -xd0)), grad_fn=ctx)
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
        yt0 = build_links(xp.sign(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.negative(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.add(xd0, xd1), grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0 = reverse_broadcast(gd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = reverse_broadcast(gd0, xd1_shape) if ctx.needs_input_grad[1] else None
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
            yt0 = build_links(xp.subtract(xd0, xd1), grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0 = reverse_broadcast(gd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = -reverse_broadcast(gd0, xd1_shape) if ctx.needs_input_grad[1] else None
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
            yt0 = build_links(xp.sin(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.cos(xd0), grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = -gd0 * xp.sin(xd0)
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
            yt0 = build_links(xp.log(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.abs(xd0), grad_fn=ctx)
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
            yt0 = build_links(xp.power(xd0, xd1), grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1, yt0)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        xd0, xd1, yd0 = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = ctx.xp
        grad0 = reverse_broadcast(gd0 * xd1 * yd0 / xd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = reverse_broadcast(gd0 * xp.log(xd0) * yd0, xd1_shape) if ctx.needs_input_grad[1] else None
        return grad0, grad1


class Mul(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if xt1.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.multiply(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = build_links(xp.multiply(xd0, xd1), grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        xd0, xd1 = ctx.saved_tensors
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
            if xt1.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.divide(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = build_links(xp.divide(xd0, xd1), grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        xd0, xd1 = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 / xd1, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad1 = reverse_broadcast(-gd0 * xd0 / (xd1 * xd1), xd1_shape)
        return grad0, grad1
