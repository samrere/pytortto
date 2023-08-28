# from .grad_fcn_generator import generate_grad_func
#
# generate_grad_func('grad_fcn_config.yaml', 'grad_fcn.py')
# del generate_grad_func

__all__ = ['helper', 'grad_mode', 'grad_fcn_misc', 'grad_fcn', 'grad_nn', 'function']

from .function import Function
