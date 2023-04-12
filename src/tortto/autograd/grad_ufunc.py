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
        return grad[0] * xp.zeros_like(grad[0])

@register_gradients(np.negative,cp.negative)
def backward(yt, grad, params):
    xt = yt.parents[0][0]
    if xt.requires_grad:
        xt.grad += -grad
@register_gradients(np.add, cp.add)
def backward(yt, grad, params):
    x0t, x1t = yt.parents[0][0],yt.parents[1][0]
    if x0t.requires_grad:
        x0t.grad += reverse_broadcast(grad, x0t.shape)
    if x1t.requires_grad:
        x1t.grad += reverse_broadcast(grad, x1t.shape)


@register_gradients(np.subtract, cp.subtract)
def backward(yt, grad, params):
    x0t, x1t = yt.parents[0][0], yt.parents[1][0]
    if x0t.requires_grad:
        x0t.grad += reverse_broadcast(grad, x0t.shape)
    if x1t.requires_grad:
        x1t.grad += -reverse_broadcast(grad, x1t.shape)


###########################################
## requires input values during backward ##
###########################################


def sin(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.sin, xt)
@inplace_precheck
def sin_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.sin, xt, out=xd, params={'copy':xd.copy()})

@register_gradients(np.sin, cp.sin)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p=yt.parents[0]
    xt= p[0]
    if xt.requires_grad:
        xd = params['copy'] if 'copy' in params else  get_data(p)
        xt.grad += grad * xp.cos(xd)


def cos(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.cos, xt)
@inplace_precheck
def cos_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.cos, xt, out=xd, params={'copy':xd.copy()})

@register_gradients(np.cos, cp.cos)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p=yt.parents[0]
    xt = p[0]
    if xt.requires_grad:
        xd = params['copy'] if 'copy' in params else  get_data(p)
        xt.grad += -grad * xp.sin(xd)


def log(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.log, xt)
@inplace_precheck
def log_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.log, xt, out=xd, params={'copy':xd.copy()})

@register_gradients(np.log, cp.log)
def backward(yt, grad, params):
    p=yt.parents[0]
    xt = p[0]
    if xt.requires_grad:
        xd = params['copy'] if 'copy' in params else  get_data(p)
        xt.grad += grad / xd


def abs(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.abs, xt)

def abs_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.abs, xt, out=xd, params={'copy': xd.copy()})

@register_gradients(np.abs, cp.abs)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p=yt.parents[0]
    xt = p[0]
    if xt.requires_grad:
        xd=params['copy'] if 'copy' in params else get_data(p)
        xt.grad += grad * xp.sign(xd)


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