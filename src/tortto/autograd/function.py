import tortto as tt
from .helper import get_data

scipy_is_loaded = bool(tt.find_spec('scipy'))
if scipy_is_loaded:
    import scipy.sparse as sci_sparse

if tt.cupy_is_loaded:
    import cupyx as cpx


class FunctionBase(object):
    __slots__ = ['variable', 'to_save', 'next_functions', 'prev_function_counts', 'needs_input_grad', 'grad', 'params',
                 'requires_grad', 'xp']

    def __init__(self):
        self.variable = None
        self.to_save = None
        self.next_functions = None
        self.prev_function_counts = 0
        self.needs_input_grad = None
        self.requires_grad = False
        self.grad = None
        self.params = None
        self.xp = None

    def save_for_backward(self, *tensors):  # inputs can be Tensor, Parameter, or None
        self.to_save = tuple(None if t is None else (t, t._version) for t in tensors)

    @property
    def saved_tensors(self):  # output tensor.data
        return tuple(get_data(pair) for pair in self.to_save)

    def clear(self):  # clear grad etc. after backward.
        self.to_save = None
        if self.__class__ is tt.AccumulateGrad or self._forward_cls is tt.ToCopy:
            self.grad = [None]
        else:
            self.grad = None
            self.params = None


class BackwardFunction(FunctionBase):  # metaclass for all grad_fn
    def apply(self, *args):
        out = self._forward_cls.backward(self, *args)  # output is xparray or tuple/list of xparray
        if out.__class__ is not tuple and out.__class__ is not list:  # grad from Cat is a list
            out = (out,)
        if len(out) != len(self.needs_input_grad):
            raise RuntimeError(f'function {self.__class__.__name__} returned an incorrect number of gradients'
                               f' (expected {len(self.needs_input_grad)}, got {len(out)})')

        ######################## backward assertion starts, can be commented out
        # for o in out:
        #     if o is None:
        #         continue
        #     if tt.cupy_is_loaded and o.__class__ is cpx.scipy.sparse._csr.csr_matrix:
        #         continue
        #     if scipy_is_loaded and o.__class__ is sci_sparse._csr.csr_matrix:
        #         continue
        #     for g in self.grad:
        #         if g is None:  # g can be None in Split
        #             continue
        #         assert o.dtype.type is g.dtype.type, f"backward dtype error at {self.__class__.__name__}, " \
        #                                              f"input grad is {g.dtype.type.__name__} " \
        #                                              f"whereas output grad is {o.dtype.type.__name__}"
        ######################## backward assertion ends, can be commented out
        return out


class AccumulateGrad(BackwardFunction):
    def apply(self, *args):
        """
        import numpy as np
        x = np.lib.stride_tricks.as_strided(1, shape=(4,), strides=[0])  # [1,1,1,1]
        x1 = x[:2] * np.array([1, 2])  # [1,2]
        x2 = x[2:]  # [1,1]
        x2 += x1  # should be [2,3]
        print(x2)  # [3,3] in numpy and [2,2] in cupy

        This is because x2 is not contiguous.
        Therefore, when self.variable.grad is None, we need to make self.grad[0] contiguous.
        Otherwise, further inplace add operation on it would give wrong result
        """
        ######################## shape assertion ends, can be commented out
        # assert self.grad[0].shape == self.variable.shape, f"backward shape error. grad shape is {self.grad[0].shape} " \
        #                                                   f"whereas variable shape is {self.variable.shape}"
        ######################## shape assertion ends, can be commented out
        if self.variable.grad is None:
            xp = tt.cp if self.variable.data.__class__ is tt.cparray else tt.np

            self.variable.grad = xp.ascontiguousarray(self.grad[0])
        else:
            self.variable.grad += self.grad[0]


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
        # cls._backward_cls = grad_fn_class
        grad_fn = grad_fn_class()  # instantiate a grad_fn object
        grad_fn.params = params

        ## check if output requires grad, as well as inplace precheck
        next_functions = []
        needs_input_grad = []
        requires_grad = False
        for i in inputs:
            i_requires_grad = False if i is None else i.requires_grad
            needs_input_grad.append(i_requires_grad)
            requires_grad |= i_requires_grad
            if i_requires_grad:
                if i.grad_fn is None:
                    # create an AccumulateGrad object and link
                    acc_grad = AccumulateGrad()
                    acc_grad.variable = i
                    acc_grad.prev_function_counts += 1
                    acc_grad.grad = [None]
                    acc_grad.next_functions = tuple()
                    next_functions.append((acc_grad, 0))
                else:
                    i.grad_fn.prev_function_counts += 1
                    next_functions.append((i.grad_fn, i._output_idx))
            else:
                next_functions.append((None, 0))
        grad_fn.needs_input_grad = tuple(needs_input_grad)
        grad_fn.next_functions = tuple(next_functions)
        grad_fn.requires_grad = requires_grad
        grad_fn.xp = tt.cp if inputs[0].data.__class__ is tt.cparray else tt.np

        ## forward
        results = cls.forward(grad_fn, *inputs, **params)
        is_tensor = False
        if results.__class__ is tt.Tensor:
            is_tensor = True
            results = (results,)

        ######################## forward assertion starts, can be commented out
        # for i in range(len(results)):
        #     r = results[i]
        #     if params.get('inplace'):
        #         if inputs[0].data_ptr() != r.data_ptr():
        #             raise RuntimeError(f"inplace is True but output address is {r.data_ptr()}, whereas "
        #                                f"address of inputs[0] is {inputs[0].data_ptr()}")
        #     assert r.data.__class__ in {tt.nparray, tt.cparray}, f'forward output of {cls.__name__} is ' \
        #                                                          f'{r.data.__class__}, not xparray'
        #
        #     for inp in inputs:
        #         if inp is None:
        #             continue
        #         if i == 1 and (cls is tt.Max0 or cls is tt.Min0):
        #             continue
        #         assert r.data.dtype.type is inp.data.dtype.type, f"forward dtype error at {cls.__name__}, " \
        #                                                          f"input is {inp.data.dtype.type.__name__} whereas " \
        #                                                          f"output is {r.data.dtype.type.__name__}"
        ######################## forward assertion ends, can be commented out

        grad_fn.grad = [None] * len(results)

        if is_tensor:
            results = results[0]
        return results
