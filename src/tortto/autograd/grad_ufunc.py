from tortto import np, cp, cparray
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
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.sqrt(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.sqrt(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 / (yd0 * 2)
        return grad0


def sqrt(input):
    return Sqrt.apply(input, inplace=False)


def sqrt_(input):
    return Sqrt.apply(input, inplace=True)


class Exp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.exp(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * yd0
        return grad0


def exp(input):
    return Exp.apply(input, inplace=False)


def exp_(input):
    return Exp.apply(input, inplace=True)


class Tan(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tan(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.tan(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * (1 + yd0 * yd0)
        return grad0


def tan(input):
    return Tan.apply(input, inplace=False)


def tan_(input):
    return Tan.apply(input, inplace=True)


class Tanh(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.tanh(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.tanh(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * (1 - yd0 * yd0)
        return grad0


def tanh(input):
    return Tanh.apply(input, inplace=False)


def tanh_(input):
    return Tanh.apply(input, inplace=True)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.exp(-xp.logaddexp(0, -xd0, out=xd0), out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.exp(-xp.logaddexp(0, -xd0)), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        yd0, = ctx.saved_tensors
        grad0 = gd0 * yd0 * (1 - yd0)
        return grad0


def sigmoid(input):
    return Sigmoid.apply(input, inplace=False)


def sigmoid_(input):
    return Sigmoid.apply(input, inplace=True)


class Sign(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        yt0 = tt.tensor(xp.sign(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xp = cp if gd0.__class__ is cparray else np
        grad0 = xp.zeros_like(gd0)
        return grad0


def sign(input):
    return Sign.apply(input)


class Neg(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.negative(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.negative(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0 = -gd0
        return grad0


def neg(input):
    return Neg.apply(input, inplace=False)


negative = neg


def neg_(input):
    return Neg.apply(input, inplace=True)


negative_ = neg


class Add(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.add(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.add(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0 = reverse_broadcast(gd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = reverse_broadcast(gd0, xd1_shape) if ctx.needs_input_grad[1] else None
        return grad0, grad1


def add(input, other):
    return Add.apply(input, other, inplace=False)


class Sub(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            xp.subtract(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
        else:
            yt0 = tt.tensor(xp.subtract(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0 = reverse_broadcast(gd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = -reverse_broadcast(gd0, xd1_shape) if ctx.needs_input_grad[1] else None
        return grad0, grad1


def sub(input, other):
    return Sub.apply(input, other, inplace=False)


subtract = sub


class Sin(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if requires_grad:
                params['copy'] = xd0.copy()
            xp.sin(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None)
        else:
            yt0 = tt.tensor(xp.sin(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = cp if gd0.__class__ is cparray else np
        grad0 = gd0 * xp.cos(xd0)
        return grad0


def sin(input):
    return Sin.apply(input, inplace=False)


def sin_(input):
    return Sin.apply(input, inplace=True)


class Cos(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if requires_grad:
                params['copy'] = xd0.copy()
            xp.cos(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None)
        else:
            yt0 = tt.tensor(xp.cos(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = cp if gd0.__class__ is cparray else np
        grad0 = -gd0 * xp.sin(xd0)
        return grad0


def cos(input):
    return Cos.apply(input, inplace=False)


def cos_(input):
    return Cos.apply(input, inplace=True)


class Log(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if requires_grad:
                params['copy'] = xd0.copy()
            xp.log(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None)
        else:
            yt0 = tt.tensor(xp.log(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
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


def log(input):
    return Log.apply(input, inplace=False)


def log_(input):
    return Log.apply(input, inplace=True)


class Abs(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if requires_grad:
                params['copy'] = xd0.copy()
            xp.abs(xd0, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None)
        else:
            yt0 = tt.tensor(xp.abs(xd0), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        xp = cp if gd0.__class__ is cparray else np
        grad0 = gd0 * xp.sign(xd0)
        return grad0


def abs(input):
    return Abs.apply(input, inplace=False)


absolute = abs


def abs_(input):
    return Abs.apply(input, inplace=True)


class Pow(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if requires_grad:
                params['copy'] = xd0.copy()
            xp.power(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None, xt1, yt0)
        else:
            yt0 = tt.tensor(xp.power(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
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
        xp = cp if gd0.__class__ is cparray else np
        grad0 = reverse_broadcast(gd0 * xd1 * yd0 / xd0, xd0_shape) if ctx.needs_input_grad[0] else None
        grad1 = reverse_broadcast(gd0 * xp.log(xd0) * yd0, xd1_shape) if ctx.needs_input_grad[1] else None
        return grad0, grad1


def pow(input, other):
    return Pow.apply(input, other, inplace=False)


class Mul(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if xt1.requires_grad:
                params['copy'] = xd0.copy()
            xp.multiply(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = tt.tensor(xp.multiply(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            xd1 = get_data(ctx.to_save[1])
            grad0 = reverse_broadcast(gd0 * xd1, xd0_shape)
        if ctx.needs_input_grad[1]:
            xd0 = ctx.params['copy'] if ctx.to_save[0] is None else get_data(ctx.to_save[0])
            grad1 = reverse_broadcast(gd0 * xd0, xd1_shape)
        return grad0, grad1


def mul(input, other):
    return Mul.apply(input, other, inplace=False)


multiply = mul


class Div(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if xt1.requires_grad:
                params['copy'] = xd0.copy()
            xp.divide(xd0, xd1, out=xd0)
            yt0 = inplace_update(xt0, requires_grad, ctx)
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = tt.tensor(xp.divide(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params['shape'] = (xd0.shape, xd1.shape)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        xd1 = get_data(ctx.to_save[1])
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 / xd1, xd0_shape)
        if ctx.needs_input_grad[1]:
            xd0 = ctx.params['copy'] if ctx.to_save[0] is None else get_data(ctx.to_save[0])
            grad1 = reverse_broadcast(-gd0 * xd0 / (xd1 * xd1), xd1_shape)
        return grad0, grad1


def div(input, other):
    return Div.apply(input, other, inplace=False)


divide = div

