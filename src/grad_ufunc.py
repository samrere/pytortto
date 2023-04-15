from tortto import np, cp, cparray
from .helper import *
from .function import *

class Sqrt(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad
        if params['inplace']:
            yd0 = xp.sqrt(xd0, out=xd0)
            yd0._version += 1
            yt0 = xt0
        else:
            yt0 = tt.tensor(xp.sqrt(xd0), requires_grad=requires_grad, copy=False, _output_idx=0)
        if requires_grad:
            yt0.grad_fn = ctx
        ctx.save_for_backward(yt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        yd0, = ctx.saved_tensors
        gd0, = grad_outputs
        xp = cp if gd0.__class__ is cparray else np
        grad0 = gd0 / xp.multiply(yd0, 2, dtype=yd0.dtype)
        return grad0
