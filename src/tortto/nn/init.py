import math

from .parameter import *

def _no_grad_uniform_(tensor, a, b):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    tensor.data = xp.random.uniform(low=a, high=b, size=tensor.shape).astype(tensor.dtype)
    return tensor

def _no_grad_normal_(tensor, mean, std):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    tensor.data = xp.random.normal(loc=mean, scale=std, size=tensor.shape).astype(tensor.dtype)
    return tensor

def _no_grad_fill_(tensor, val):
    tensor.data.fill(val)
    return tensor


def ones_(tensor):
    return _no_grad_fill_(tensor, 1.)


def zeros_(tensor):
    return _no_grad_fill_(tensor, 0.)

def constant_(tensor, val):
    return _no_grad_fill_(tensor, val)

def normal_(tensor: Tensor, mean = 0., std = 1.):
    return _no_grad_normal_(tensor, mean, std)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor.data[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    xp = cp if tensor.data.__class__ is cp_ndarray else np
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    tensor.data = xp.random.uniform(low=-bound, high=bound, size=tensor.shape).astype(tensor.dtype)
    return tensor

def xavier_uniform_(tensor, gain = 1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)

def xavier_normal_(tensor, gain = 1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std)