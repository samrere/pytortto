from tortto import np, cp, cparray
from .helper import *
from .function import *
"""
t means tensor object: xt, yt are tensors.
d means tensor data: xd, yd are .data attribute of xt and yt.
"""

""" checklist
1**. use @inplace_precheck before inplace ops
2**. for inputs that can be numpy scalar, use np.add, np.multiply etc. (np.array(1, dtype=np.float32) + 2 will be float64)
3*. use y if possible in backward, to avoid recalculation
4*. be careful of overflow in np.exp 
"""
#####################################
## requires output during backward ##
#####################################
class Sqrt(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd = inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.sqrt(xd, out=xd)
        else:
            result = xp.sqrt(xd)
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, *grad):
        yd=ctx.saved_tensors[0]
        xp = cp if grad[0].__class__ is cparray else np
        return grad[0]*xp.multiply(yd, 0.5, dtype=yd.dtype)
def sqrt(xt):
    return Sqrt.apply(xt, inplace=False)
def sqrt_(xt):
    return Sqrt.apply(xt, inplace=True)

class Exp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.exp(xd, out=xd)
        else:
            result = xp.exp(xd)
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, *grad):
        yd=ctx.saved_tensors[0]
        return grad[0] * yd
def exp(xt):
    return Exp.apply(xt, inplace=False)
def exp_(xt):
    return Exp.apply(xt, inplace=True)

class Tan(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.tan(xd, out=xd)
        else:
            result = xp.tan(xd)
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        yd=ctx.saved_tensors[0]
        return grad[0] * xp.add(1, yd * yd, dtype=yd.dtype)
def tan(xt):
    return Tan.apply(xt, inplace=False)
def tan_(xt):
    return Tan.apply(xt, inplace=True)

class Tanh(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.tanh(xd, out=xd)
        else:
            result = xp.tanh(xd)
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        yd=ctx.saved_tensors[0]
        return grad[0] * xp.subtract(1, yd * yd, dtype=yd.dtype)
def tanh(xt):
    return Tanh.apply(xt, inplace=False)
def tanh_(xt):
    return Tanh.apply(xt, inplace=True)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.exp(-xp.logaddexp(0,-xd, out=xd), out=xd)
        else:
            result = xp.exp(-xp.logaddexp(0,-xd))
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        yd=ctx.saved_tensors[0]
        return grad[0] * yd * xp.subtract(1, yd, dtype=yd.dtype)
def sigmoid(xt):
    return Sigmoid.apply(xt, inplace=False)
def sigmoid_(xt):
    return Sigmoid.apply(xt, inplace=True)

######################################
## requires nothing during backward ##
######################################
class Sign(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd = inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        result = xp.sign(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        return xp.zeros_like(grad[0])
def sign(xt):
    return Sign.apply(xt)

class Neg(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.negative(xd, out=xd)
        else:
            result = xp.negative(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        return -grad[0]
def neg(xt):
    return Neg.apply(xt, inplace=False)
negative = neg
def neg_(xt):
    return Neg.apply(xt, inplace=True)
negative_ = neg_

class Add(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd0, xd1 = inputs[0].data, inputs[1].data
        xp = cp if xd0.__class__ is cparray else np
        if params['inplace']:
            result = xp.add(xd0, xd1, out=xd0)
        else:
            result = xp.add(xd0, xd1)
        ctx.params={'shape':(xd0.shape, xd1.shape)}
        return result
    @staticmethod
    def backward(ctx, *grad):
        grad0, grad1= None, None
        xd0_shape, xd1_shape = ctx.params['shape']
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(grad, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad0 = reverse_broadcast(grad, xd1_shape)
        return grad0, grad1

def add(input, other):
    return Add.apply(input, other, inplace=False)



class Sub(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd0, xd1 = inputs[0].data, inputs[1].data
        xp = cp if xd0.__class__ is cparray else np
        if params['inplace']:
            result = xp.subtract(xd0, xd1, out=xd0)
        else:
            result = xp.subtract(xd0, xd1)
        ctx.params={'shape':(xd0.shape, xd1.shape)}
        return result
    @staticmethod
    def backward(ctx, *grad):
        grad0, grad1= None, None
        xd0_shape, xd1_shape = ctx.params['shape']
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(grad, xd0_shape)
        if ctx.needs_input_grad[1]:
            grad0 = -reverse_broadcast(grad, xd1_shape)
        return grad0, grad1

def sub(input, other):
    return Sub.apply(input, other, inplace=False)
subtract = sub

###########################################
## requires input values during backward ##
###########################################
class Sin(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.sin(xd, out=xd)
            ctx.params = {'copy': xd.copy()}
        else:
            result = xp.sin(xd)
            ctx.save_for_backward(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        xd = ctx.params['copy'] if 'copy' in ctx.params else ctx.saved_tensors[0]
        return grad[0] * xp.cos(xd)
def sin(xt):
    return Sin.apply(xt, inplace=False)
def sin_(xt):
    return Sin.apply(xt, inplace=True)


class Cos(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.cos(xd, out=xd)
            ctx.params = {'copy': xd.copy()}
        else:
            result = xp.cos(xd)
            ctx.save_for_backward(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        xd = ctx.params['copy'] if 'copy' in ctx.params else ctx.saved_tensors[0]
        return -grad[0] * xp.sin(xd)
def cos(xt):
    return Cos.apply(xt, inplace=False)
def cos_(xt):
    return Cos.apply(xt, inplace=True)


class Log(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd=inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.log(xd, out=xd)
            ctx.params = {'copy': xd.copy()}
        else:
            result = xp.log(xd)
            ctx.save_for_backward(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xd = ctx.params['copy'] if 'copy' in ctx.params else ctx.saved_tensors[0]
        return grad[0] / xd
def log(xt):
    return Log.apply(xt, inplace=False)
def log_(xt):
    return Log.apply(xt, inplace=True)


class Abs(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xd = inputs[0].data
        xp = cp if xd.__class__ is cparray else np
        if params['inplace']:
            result = xp.abs(xd, out=xd)
            ctx.params = {'copy': xd.copy()}
        else:
            result = xp.abs(xd)
            ctx.save_for_backward(xd)
        return result
    @staticmethod
    def backward(ctx, *grad):
        xp = cp if grad[0].__class__ is cparray else np
        xd = ctx.params['copy'] if 'copy' in ctx.params else ctx.saved_tensors[0]
        return grad[0] * xp.sign(xd)
def abs(xt):
    return Abs.apply(xt, inplace=False)
absolute = abs
def abs_(xt):
    return Abs.apply(xt, inplace=True)




######################################################################################################################
@register_gradients(np.multiply, cp.multiply)
def backward(yt, grad, params):
    p0,p1 = yt.parents[0],yt.parents[1]
    xt0 = p0[0]
    xt1 = p1[0]
    if xt0.requires_grad:
        xd1=get_data(p1)
        xt0.grad += reverse_broadcast(grad * xd1, xt0.shape)
    if xt1.requires_grad:
        xd0=params['copy'] if 'copy' in params else get_data(p0)
        xt1.grad += reverse_broadcast(grad * xd0, xt1.shape)


@register_gradients(np.divide, cp.divide)
def backward(yt, grad, params):
    p0, p1 = yt.parents[0], yt.parents[1]
    xt0 = p0[0] # numerator
    xt1 = p1[0] # denominator
    xd1 = get_data(p1)
    if xt0.requires_grad:
        xt0.grad += reverse_broadcast(grad / xd1, xt0.shape)
    if xt1.requires_grad:
        xd0 = params['copy'] if 'copy' in params else get_data(p0)
        xt1.grad += reverse_broadcast(-grad * xd0 / (xd1 * xd1), xt1.shape)


@register_gradients(np.power, cp.power)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cparray else np
    inputs = tensor.parents
    base = inputs[0]
    power = inputs[1]
    if base.requires_grad:
        base.grad += grad * power.data * base.data ** xp.subtract(power.data, 1, dtype=grad.dtype)
    if power.requires_grad:
        power.grad += grad * xp.log(base.data) * tensor.data