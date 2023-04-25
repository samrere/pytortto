import tortto as tt
from .grad_mode import *
from .helper import get_data


class FunctionBase(object):
    __slots__ = ['variable','to_save', 'next_functions','prev_function_counts','needs_input_grad','grad','params',
                 ]
    def __init__(self):
        self.variable = None
        self.to_save = None
        self.next_functions = None
        self.prev_function_counts = 0
        self.needs_input_grad = None
        self.grad = None
        self.params = None

    def save_for_backward(self, *tensors):
        self.to_save = tuple((t, t._version) if t.__class__ is tt.Tensor else None for t in tensors)

    @property
    def saved_tensors(self): # output tensor.data
        return tuple(get_data(pair) for pair in self.to_save)

    def clear(self): # clear grad etc. after backward
        self.to_save = None
        self.grad = None
        self.params = None



class BackwardFunction(FunctionBase): # metaclass for all grad_fn
    def apply(self, *args):
        out=self._forward_cls.backward(self, *args) # output is xparray or tuple/list of xparray
        if out.__class__ is not tuple and out.__class__ is not list: # grad from Cat is a list
            out=(out,)
        if len(out)!=len(self.needs_input_grad):
            raise RuntimeError(f'function {self.__class__.__name__} returned an incorrect number of gradients'
                               f' (expected {len(self.needs_input_grad)}, got {len(out)})')
        ## backward assertion, commented out
        for o in out:
            if o is not None:
                assert o.__class__ in {tt.cparray, tt.nparray}, f"backward output of {self.__class__.__name__} " \
                                                                f"is {o.__class__}, not xparray"
                for g in self.grad:
                    assert o.dtype.type is g.dtype.type, f"backward dtype error at {self.__class__.__name__}, " \
                                                     f"input grad is {g.dtype.type.__name__} " \
                                                     f"whereas output grad is {o.dtype.type.__name__}"
        return out

class AccumulateGrad(BackwardFunction):
    def apply(self, *args):
        if self.variable.grad is None:
            self.variable.grad = self.grad[0]
        else:
            self.variable.grad+=self.grad[0]

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
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("You must implement the backward method for "
                                  "your custom autograd.Function to use it with backward "
                                  "mode automatic differentiation.")

    @classmethod
    def apply(cls, *inputs, **params):
        ## create the grad_fn class
        grad_fn_class = type(cls.__name__ + 'Backward', (BackwardFunction,), {'_forward_cls': cls})
        cls._backward_cls = grad_fn_class
        grad_fn=grad_fn_class() # instantiate a grad_fn object
        grad_fn.params = params

        ## check if output requires grad, as well as inplace precheck
        next_functions = []
        needs_input_grad = []
        for i in inputs:
            needs_input_grad.append(i.requires_grad)
            if i.requires_grad:
                if i.grad_fn is None:
                    # create an AccumulateGrad object and link
                    acc_grad = AccumulateGrad()
                    acc_grad.variable = i
                    acc_grad.prev_function_counts+=1
                    acc_grad.grad = [tt._int_zero]
                    acc_grad.next_functions=tuple()
                    next_functions.append((acc_grad, 0))
                else:
                    i.grad_fn.prev_function_counts+=1
                    next_functions.append((i.grad_fn, i._output_idx))
            else:
                next_functions.append((None,0))
        grad_fn.needs_input_grad = tuple(needs_input_grad)
        grad_fn.next_functions=tuple(next_functions)

        ## forward
        results = cls.forward(grad_fn, *inputs, **params)

        is_tensor=False
        if results.__class__ is tt.Tensor:
            is_tensor=True
            results = (results,)

        ## forward assertion, commented out
        for r in results:
            if params.get('inplace'):
                if inputs[0].data_ptr()!=r.data_ptr():
                    raise RuntimeError(f"inplace is True but output address is {r.data_ptr()}, whereas "
                                       f"address of inputs[0] is {inputs[0].data_ptr()}")
            assert r.data.__class__ in {tt.nparray, tt.cparray}, f'forward output of {cls.__name__} is ' \
                                                                 f'{r.data.__class__}, not xparray'
            for i in inputs:
                assert r.data.dtype.type is i.data.dtype.type, f"forward dtype error at {cls.__name__}, " \
                                                               f"input is {i.data.dtype.type.__name__} whereas " \
                                                               f"output is {r.data.dtype.type.__name__}"


        grad_fn.grad = [tt._int_zero] * len(results)

        if not is_grad_enabled(): # if using tortto.no_grad(), disable requires_grad
            for r in results:
                r.requires_grad=False
                r.grad_fn=None

        if is_tensor:
            results = results[0]
        return results
