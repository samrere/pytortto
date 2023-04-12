import tortto as tt
from .grad_mode import *
from .helper import get_data
import tortto.autograd as au

class FunctionBase(object):
    __slots__ = ['variable','to_save', 'next_functions', 'needs_input_grad','grad','params']
    def __init__(self):
        self.variable = None
        self.to_save = [] # tuple of (tensor,version) pairs
        self.next_functions = None
        self.needs_input_grad = []
        self.grad=None
        self.params=None

    def save_for_backward(self, *tensors):
        for t in tensors:
            self.to_save.append((t, t._version))

    @property
    def saved_tensors(self): # output tensor.data
        return tuple(get_data(pair) for pair in self.to_save)

    def clear(self): # clear grad etc. after backward
        ...


class BackwardFunction(FunctionBase): # metaclass for all grad_fn
    def apply(self, *args):
        out=self._forward_cls.backward(self, *args)
        if out.__class__ is not tuple:
            out=(out,)
        if len(out)!=len(self.needs_input_grad):
            raise RuntimeError(f'function {self.__class__.__name__} returned an incorrect number of gradients'
                               f' (expected {len(self.needs_input_grad)}, got {len(out)})')
        if len(out)==1:
            out=out[0]
        return out

class AccumulateGrad(BackwardFunction):
    def apply(self, *args):
        self.variable.grad+=self.grad


class Function(FunctionBase):
    def __init__(self, *args, **kwargs):
        cls = self.__class__
        raise RuntimeError(f"{cls} should not be instantiated. Methods on autograd functions"
                      "are all static, so you should invoke them on the class itself.")

    @staticmethod
    def forward(ctx, *inputs, **params):
        raise NotImplementedError("You must implement the forward function for custom"
                                  " autograd.Function.")

    @staticmethod
    def backward(ctx, *grad):
        raise NotImplementedError("You must implement the backward method for "
                                  "your custom autograd.Function to use it with backward "
                                  "mode automatic differentiation.")

    @classmethod
    def apply(cls, *inputs, **params):
        ## create the grad_fn class
        grad_fn_class = type(cls.__name__ + 'Backward', (BackwardFunction,), {'_forward_cls': cls})
        cls._backward_cls = grad_fn_class
        grad_fn=grad_fn_class() # instantiate a grad_fn object
        grad_fn.params=params

        ## check if output requires grad, as well as inplace precheck
        requires_grad = False
        next_functions = []
        for i in inputs:
            if i.__class__ is not tt.Tensor: # can be commented out
                raise RuntimeError(f'BUG: input is not tensor at {grad_fn_class}')
            grad_fn.needs_input_grad.append(i.requires_grad)
            if i.requires_grad:
                requires_grad = True
                if i.grad_fn is None:
                    if params.get('inplace') is True:
                        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
                    else:
                        # create an AccumulateGrad object and link
                        acc_grad=AccumulateGrad()
                        acc_grad.variable=i
                        next_functions.append((acc_grad,0))
                else:
                    next_functions.append((i.grad_fn, i._output_idx))
            else:
                next_functions.append((None,0))
        grad_fn.next_functions=tuple(next_functions)


        ## if using tortto.no_grad(), disable requires_grad
        if not is_grad_enabled():
            requires_grad = False

        ## forward
        values = cls.forward(grad_fn, *inputs, **params)
        if params['inplace'] is True: ## For now, inplace is true only when there's a single output
            values._version+=1
        if values.__class__ is not tuple:
            values=(values,) # values is a tuple of xparray objects
        grad_fn.grad=(tt._int_zero,)*len(values)

        ## forward check, comment it out. do test on individual functions
        # if cls is not au.grad_fcn._cuda and cls is not au.grad_fcn._cpu:
        #     for v in values:
        #         for i in inputs:
        #             if i is not None:
        #                 assert v.dtype == i.dtype, \
        #                     f'dtype assertion error during forward at {cls.__name__}: ' \
        #                     f'value is {v.dtype} whereas input is {i.dtype}'
        #                 assert hasattr(v.data, 'device') == hasattr(i.data, 'device'), \
        #                     f'array class assertion error during forward at {cls.__name__}: ' \
        #                     f'value is {v.data.__class__} whereas input is {i.data.__class__}'

        ## create output
        outputs=[]
        for idx, v in enumerate(values):
            out = tt.tensor(v, requires_grad=requires_grad, copy=False, _output_idx=idx, _output_version=v._version)
            if requires_grad:
                out.grad_fn=grad_fn
            outputs.append(out)

        ## output
        if len(outputs)==1:
            outputs=outputs[0]
        return outputs
