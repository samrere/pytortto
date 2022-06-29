from tortto import np, cp, cp_ndarray
from .helper import *


def sin(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return compute_ufunc(xp.sin, x)


@register_gradients(np.sin, cp.sin)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        inputs[0].grad += grad * xp.cos(x)


def cos(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return compute_ufunc(xp.cos, x)


@register_gradients(np.cos, cp.cos)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = inputs[0].data
        inputs[0].grad += -grad * xp.sin(x)



def exp(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return compute_ufunc(xp.exp, x)


@register_gradients(np.exp, cp.exp)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad * tensor.data



def log(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return compute_ufunc(xp.log, x)


@register_gradients(np.log, cp.log)
def backward(tensor, grad, params):
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += grad / inputs[0].data



def tanh(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    return compute_ufunc(xp.tanh, x)


@register_gradients(np.tanh, cp.tanh)
def backward(tensor, grad, params):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # tanh'(x) = 1 - tanh^2(x)
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = tensor.data
        inputs[0].grad += grad * xp.subtract(1,  x * x, dtype=x.dtype)


def sigmoid(x):
    xp = cp if x.data.__class__ is cp_ndarray else np
    x_data = x.data
    dtype = x_data.dtype
    value = xp.divide(1, xp.add(1, xp.exp(-x_data), dtype=dtype), dtype=dtype)
    output = build_links(value, x.requires_grad, sigmoid, x)
    return output


@register_gradients(sigmoid)
def backward(tensor, grad, params):
    # sig(x) = 1 / (1 + exp(-x))
    # sig'(x) = sig(x) * (1 - sig(x))
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    if inputs[0].requires_grad:
        x = tensor.data
        inputs[0].grad += grad * x * xp.subtract(1, x, dtype=grad.dtype)

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
    xp = cp if grad.__class__ is cp_ndarray else np
    inputs = tensor.parents
    base = inputs[0]
    power = inputs[1]
    if base.requires_grad:
        base.grad += grad * power.data * base.data ** xp.subtract(power.data, 1, dtype=grad.dtype)
    if power.requires_grad:
        power.grad += grad * xp.log(base.data) * tensor.data


@register_gradients(np.negative,cp.negative)
def backward(tensor, grad, params):
    inputs = tensor.parents
    if inputs[0].requires_grad:
        inputs[0].grad += -grad


@register_gradients(np.add, cp.add)
def backward(tensor, grad, params):
    inputs = tensor.parents
    inpt0 = inputs[0]
    inpt1 = inputs[1]
    if inpt0.requires_grad:
        inpt0.grad += reverse_broadcast(grad, inpt0.shape)
    if inpt1.requires_grad:
        inpt1.grad += reverse_broadcast(grad, inpt1.shape)


@register_gradients(np.subtract, cp.subtract)
def backward(tensor, grad, params):
    inputs = tensor.parents
    inpt0 = inputs[0]
    inpt1 = inputs[1]
    if inpt0.requires_grad:
        inpt0.grad += reverse_broadcast(grad, inpt0.shape)
    if inpt1.requires_grad:
        inpt1.grad += -reverse_broadcast(grad, inpt1.shape)
