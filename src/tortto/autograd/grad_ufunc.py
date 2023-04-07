from tortto import np, cp, cparray
from .helper import *
"""
t means tensor object: xt, yt are tensors.
d means tensor data: xd, yd are .data attribute of xt and yt.
"""
#####################################
## requires output during backward ##
#####################################
def sqrt(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.sqrt, xt)
@inplace_precheck
def sqrt_(xt):
    xd=xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.sqrt, xt, out=xd)

@register_gradients(np.sqrt, cp.sqrt)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p0 = yt.parents[0]
    xt = p0.tensor
    if xt.requires_grad:
        yd = yt.myself.get_data()
        xt.grad += grad * xp.multiply(yd, 0.5, dtype=yd.dtype)

def exp(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.exp, xt)
@inplace_precheck
def exp_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.exp, xt, out=xd)

@register_gradients(np.exp, cp.exp)
def backward(yt, grad, params):
    p0 = yt.parents[0]
    xt=p0.tensor
    if xt.requires_grad:
        yd = yt.myself.get_data()
        xt.grad += grad * yd


def tan(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.tan, xt)
@inplace_precheck
def tan_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.tan, xt, out=xd)

@register_gradients(np.tan, cp.tan)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p0 = yt.parents[0]
    xt=p0.tensor
    if xt.requires_grad:
        yd = yt.myself.get_data()
        xt.grad += grad * xp.add(1, yd * yd, dtype=yd.dtype)

def tanh(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.tanh, xt)

@inplace_precheck
def tanh_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    return compute_ufunc(xp.tanh, xt, out=xd)

@register_gradients(np.tanh, cp.tanh)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p0 = yt.parents[0]
    xt=p0.tensor
    if xt.requires_grad:
        yd = yt.myself.get_data()
        xt.grad += grad * xp.subtract(1, yd * yd, dtype=yd.dtype)


def sigmoid(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    value=xp.exp(-xp.logaddexp(0,-xd))
    output = build_links(value, xt.requires_grad, sigmoid, xt)
    return output

@inplace_precheck
def sigmoid_(xt):
    xd = xt.data
    xp = cp if xd.__class__ is cparray else np
    value = xp.exp(-xp.logaddexp(0,-xd, out=xd), out=xd)
    output = build_links(value, xt.requires_grad, sigmoid, xt)
    return output


@register_gradients(sigmoid)
def backward(yt, grad, params):
    xp = cp if grad.__class__ is cparray else np
    p0 = yt.parents[0]
    xt=p0.tensor
    if xt.requires_grad:
        yd = yt.myself.get_data()
        xt.grad += grad * yd * xp.subtract(1, yd, dtype=grad.dtype)

###################################
## requires none during backward ##
###################################
@register_gradients(np.negative,cp.negative)
def backward(yt, grad, params):
    xt = yt.parents[ind][0][0]
    if xt.requires_grad:
        xt.grad += -grad
@register_gradients(np.add, cp.add)
def backward(yt, grad, params):
    parent_group=yt.parents[ind]
    x0t, x1t = parent_group[0][0],parent_group[1][0]
    if x0t.requires_grad:
        x0t.grad += reverse_broadcast(grad, x0t.shape)
    if x1t.requires_grad:
        x1t.grad += reverse_broadcast(grad, x1t.shape)


@register_gradients(np.subtract, cp.subtract)
def backward(yt, grad, params):
    parent_group = yt.parents[ind]
    x0t, x1t = parent_group[0][0], parent_group[1][0]
    if x0t.requires_grad:
        x0t.grad += reverse_broadcast(grad, x0t.shape)
    if x1t.requires_grad:
        x1t.grad += -reverse_broadcast(grad, x1t.shape)


###########################################
## requires input values during backward ##
###########################################


def sin(x):
    xp = cp if x.data.__class__ is cparray else np
    return compute_ufunc(xp.sin, x)


@register_gradients(np.sin, cp.sin)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cparray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        inputs[0].grad += grad * xp.cos(x)


def cos(x):
    xp = cp if x.data.__class__ is cparray else np
    return compute_ufunc(xp.cos, x)


@register_gradients(np.cos, cp.cos)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cparray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        inputs[0].grad += -grad * xp.sin(x)

def log(x):
    xp = cp if x.data.__class__ is cparray else np
    return compute_ufunc(xp.log, x)


@register_gradients(np.log, cp.log)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cparray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad / inputs[0].data


def abs(x):
    pass


@register_gradients(np.abs, cp.abs)
def backward(tensor, grad, params):
    pass




@register_gradients(np.multiply, cp.multiply)
def backward(tensor, grad, params):
    # element-wise multiplication
    inputs = tensor.parents
    inpt0 = inputs[0]
    inpt1 = inputs[1]
    if inpt0.requires_grad:
        inpt0.grad += reverse_broadcast(grad * inpt1.data, inpt0.shape)
    if inpt1.requires_grad:
        inpt1.grad += reverse_broadcast(grad * inpt0.data, inpt1.shape)


@register_gradients(np.divide, cp.divide)
def backward(tensor, grad, params):
    inputs = tensor.parents
    num = inputs[0]
    deno = inputs[1]
    if num.requires_grad:
        num.grad += reverse_broadcast(grad / deno.data, num.shape)
    if deno.requires_grad:
        deno.grad += reverse_broadcast(-grad * num.data / (deno.data * deno.data), deno.shape)


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