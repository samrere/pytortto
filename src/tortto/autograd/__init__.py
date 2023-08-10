from .grad_ufunc_generator import generate_grad_ufunc
generate_grad_ufunc()
del generate_grad_ufunc
__all__=['helper','grad_mode','grad_fcn','grad_ufunc','grad_nn','function']
from .function import Function

