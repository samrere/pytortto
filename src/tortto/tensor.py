import tortto
from tortto import np, cp, cparray, nparray, cupy_is_loaded
from .VariableFunctions import *

int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128

Number = {int, float, bool}
default_dtype = {int64, float32, complex64, np.bool_}


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None, copy=True, **kwargs):
        if dtype is None: # all defaults to float32
            dtype=float32
        if cupy_is_loaded and (data.__class__ is cparray or data.__class__ is cp.ndarray):
            self.data = cparray(data, dtype=dtype, copy=copy)
        else:
            self.data = nparray(data, dtype=dtype, copy=copy)
        self.grad = None
        self.grad_fn = kwargs.get('grad_fn')
        self.requires_grad = requires_grad

        self._output_idx = kwargs.get('_output_idx')

    ################
    ## properties ##
    ################
    @property
    def _version(self):
        return self.data._version[0]
    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        if val.__class__ is not bool:
            raise RuntimeError('requires_grad must be a bool')
        if val is True:
            dtype_type = self.data.dtype.type
            if not issubclass(dtype_type, np.complexfloating) and not issubclass(dtype_type, np.floating):
                raise RuntimeError('only Tensors of floating point and complex dtype can require gradients')
        self._requires_grad = val

    @property
    def T(self):
        if self.ndim!=2:
            raise RuntimeError(f"x.T expects a tensor with 2 dimensions, but self is {self.ndim}D")
        return permute(self, (1,0))

    @property
    def device(self):
        if self.data.__class__ is cparray:
            return self.data.device
        else:
            return 'cpu'

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @shape.setter
    def shape(self, value):
        self.data.shape = value

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def itemsize(self):
        return self.data.itemsize

    @property
    def strides(self):
        return self.data.strides

    @strides.setter
    def strides(self, value):
        self.data.strides = value


    ####################################
    ## operator overload (no grad fn) ##
    ####################################
    def __len__(self):
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __repr__(self):
        device = f", device='{self.device}'" if self.device != 'cpu' else ''
        dtype = f', dtype={self.dtype}' if self.data is not None and self.dtype.type not in default_dtype else ''
        grad_fn = f', grad_fn=<{self.grad_fn.__class__.__name__}>' if self.grad_fn else ''
        requires_grad = f', requires_grad=True' if self.requires_grad and not self.grad_fn else ''
        if self.data is None:
            s = str(None)
        else:
            s = np.array2string(self.data, separator=', ', precision=4).replace('\n', '\n' + ' ' * 7)
        return f'tensor({s}{device}{dtype}{grad_fn}{requires_grad})'
    def __array__(self, dtype=None):
        """
        useful when setting tensor values using another tensor:

        import tortto as tt
        x=tt.randn(2,4)
        x[:,::2]=tt.zeros((2,2))

        https://numpy.org/devdocs/user/basics.dispatch.html
        """
        return self.data.astype(dtype, copy=False)


    def __eq__(self, other):
        # use default dtype (float32) not bool, because any further operations on bool will result in float64:
        # tt.tensor([True, False], dtype=bool).mean() --> float64
        if other.__class__ is Tensor:
            other=other.data
        elif other.__class__ not in Number:
            return False
        return tt.tensor(self.data == other, dtype=float32, copy=False)

    def __ne__(self, other):
        if other.__class__ is Tensor:
            other = other.data
        elif other.__class__ not in Number:
            return True
        return tt.tensor(self.data != other, dtype=float32, copy=False)
    def __lt__(self, other):
        if other.__class__ is Tensor:
            other=other.data
        elif other.__class__ not in Number:
            raise TypeError(f"'<' not supported between instances of '{self.__class__}' and '{other.__class__}'")
        return tt.tensor(self.data < other, dtype=float32, copy=False)
    def __le__(self, other):
        if other.__class__ is Tensor:
            other=other.data
        elif other.__class__ not in Number:
            raise TypeError(f"'<=' not supported between instances of '{self.__class__}' and '{other.__class__}'")
        return tt.tensor(self.data <= other, dtype=float32, copy=False)

    def __gt__(self, other):
        if other.__class__ is Tensor:
            other=other.data
        elif other.__class__ not in Number:
            raise TypeError(f"'>' not supported between instances of '{self.__class__}' and '{other.__class__}'")
        return tt.tensor(self.data > other, dtype=float32, copy=False)
    def __ge__(self, other):
        if other.__class__ is Tensor:
            other=other.data
        elif other.__class__ not in Number:
            raise TypeError(f"'>=' not supported between instances of '{self.__class__}' and '{other.__class__}'")
        return tt.tensor(self.data >= other, dtype=float32, copy=False)

    #####################################
    ## operator overload (has grad fn) ##
    #####################################
    def __neg__(self):
        return neg(self)

    def __add__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return add(self, other)

    __radd__ = __add__

    def __iadd__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return Add.apply(self, other, inplace=True)

    add_ = __iadd__
    add=__add__

    def __sub__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return sub(self, other)

    # @inplace_precheck
    def __isub__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return Sub.apply(self, other, inplace=True)

    subtract_=__isub__
    sub_=__isub__
    subtract = __sub__
    sub = __sub__

    def __rsub__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return sub(other, self)


    def __mul__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return mul(self, other)

    __rmul__ = __mul__

    def __imul__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return Mul.apply(self, other, inplace=True)

    multiply_ = __imul__
    mul_=__imul__
    multiply = __mul__
    mul = __mul__

    def __truediv__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return div(self, other)

    def __idiv__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return Div.apply(self, other, inplace=True)

    divide_ = __idiv__
    div_=__idiv__
    divide = __truediv__
    div = __truediv__
    def __rtruediv__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return div(other, self)

    def __pow__(self, other):  # i.e. Tensor**3
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return pow(self, other)

    def __rpow__(self, other):  # i.e. 3**Tensor
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return pow(other, self)

    def __ipow__(self, other):
        xp = cp if self.data.__class__ is cparray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return Pow.apply(self, other, inplace=True)

    pow = __pow__
    pow_ = __ipow__

    def __matmul__(self, other):
        return matmul(self, other)
    def __imatmul__(self, other):
        raise RuntimeError(f"In-place matmul is not supported. Use 'a = a @ b' instead of 'a @= b'.")

    def __getitem__(self, key):
        return Slice.apply(self, key=key)


    def __setitem__(self, key, value):
        if value.__class__ is not tt.Tensor:
            value = tt.tensor(value, copy=False)
        if key.__class__ is tt.Tensor:
            key=key.data
        return CopySlices.apply(self, value, key=key)

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            shape = self.shape
            if dim < -len(shape) or dim > len(shape) - 1:
                raise IndexError(
                    f'Dimension out of range (expected to be in range of [-{len(shape)}, {len(shape) - 1}], but got {dim})')
            return self.shape[dim]

    def numel(self):
        # Returns the total number of elements in the input tensor.
        return self.data.size


    def item(self):
        return self.data.item()

    def data_ptr(self):
        # https://docs.cupy.dev/en/latest/user_guide/interoperability.html#device-memory-pointers
        return self.data.data.ptr if self.data.__class__ is cparray else self.data.ctypes.data

    def cuda(self):
        return ToCopy.apply(self, target_device='cuda')

    def cpu(self):
        return ToCopy.apply(self, target_device='cpu')

    def detach(self):
        # same as pytorch, where detached tensor share the same data with the original one.
        return Tensor(self.data, requires_grad=False, dtype=self.data.dtype, copy=False)

    def numpy(self):
        data = self.data
        if data.__class__ is cparray:
            raise TypeError(
                f"can't convert {self.device} device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return data

    def contiguous(self):
        """
        different from pytorch: tortto `contiguous` does it inplace.
        """
        xp = cp if self.data.__class__ is cparray else np
        self.data[...] = xp.ascontiguousarray(self.data)
        return self

    def is_contiguous(self):
        return self.data.flags['C_CONTIGUOUS']

    def copy_(self, array): # check this later!
        """
        1. the input can be tensor or array instead of tensor only as in pytorch
        2. used in module _load_from_state_dict. copy numpy array from checkpoint to tensor
           array from checkpoint is always numpy array.
        """
        if array.__class__ is not Tensor:
            array=tt.tensor(array, copy=False)
        return Copy.apply(self, array)



    def sum(self, dim=None, keepdim=False):
        return sum(self, dim, keepdim)

    def mean(self, dim=None, keepdim=False):
        return mean(self, dim, keepdim)

    def var(self, dim=None, unbiased=True, keepdim=False):
        return var(self, dim, unbiased, keepdim)



    def view(self, *shape):
        if shape[0].__class__ is not int:
            shape = shape[0]
        return View.apply(self, shape=shape)

    def flatten(self, start_dim=0, end_dim=-1):
        return flatten(self, start_dim, end_dim)

    def permute(self, dims):
        return permute(self, dims)
    def transpose(self, dim0, dim1):
        return transpose(self, dim0, dim1)

    swapaxes=transpose
    swapdims=transpose

    def moveaxis(self, source, destination):
        return moveaxis(self, source, destination)

    def chunk(self, chunks, dim=0):
        return chunk(self, chunks, dim=dim)

    def split(self, split_size_or_sections, dim=0):
        return split(self, split_size_or_sections, dim=dim)

    def repeat(self, *sizes):
        if sizes[0].__class__ is tuple:
            sizes=sizes[0]
        return Repeat.apply(self, sizes=sizes)

    def expand(self, *sizes):
        if sizes[0].__class__ is tuple:
            sizes=sizes[0]
        return Expand.apply(self, sizes=sizes)

    def squeeze(self, dim=None):
        return squeeze(self, dim=dim)

    def unsqueeze(self, dim):
        return unsqueeze(self, dim)

    def logical_or(self, x2):
        return logical_or(self, x2)

    def logical_and(self, x2):
        return logical_and(self, x2)

    def logical_not(self):
        return logical_not(self)

    def logical_xor(self, x2):
        return logical_xor(self, x2)

    def masked_fill(self, mask, val):
        return masked_fill(self, mask, val)

    def masked_fill_(self, mask, val):
        return masked_fill_(self, mask, val)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim, keepdim)

    def argmin(self, dim=None, keepdim=False):
        return argmin(self, dim, keepdim)

    def sin(self):
        return sin(self)

    def cos(self):
        return cos(self)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def tanh(self):
        return tanh(self)

    def sigmoid(self):
        return sigmoid(self)

    def softmax(self,dim):
        return tortto.nn.functional.softmax(self,dim)

    def log_softmax(self,dim):
        return tortto.nn.functional.log_softmax(self,dim)

    def type(self, dtype):
        self.data = self.data.astype(dtype, copy=False)
        return self
    # no need for inplace check, as it calls __setitem__
    def normal_(self, mean=0, std=1):
        xp = cp if self.data.__class__ is cparray else np
        self[...]=xp.random.normal(loc=mean, scale=std, size=self.shape)

    def uniform_(self, low=0, high=1):
        xp=cp if self.data.__class__ is cparray else np
        self[...]=xp.random.uniform(low=low, high=high, size=self.shape)
    def fill_(self,value):
        self[...] = value

    def backward(self, gradient=None):
        if not self.requires_grad:
            assert self.grad_fn is None, 'bug!'
            raise RuntimeError('element 0 of tensors does not require grad and does not have a grad_fn')

        xd=self.data
        xp = cp if xd.__class__ is cparray else np
        if gradient is None:
            gradient = xp.expand_dims(xp.array(1, dtype=xd.dtype), axis=tuple(range(xd.ndim)))

        elif isinstance(gradient, Tensor):
            if gradient.device!=self.device:
                raise RuntimeError(f"invalid gradient at index 0 - expected device {self.device} but got {gradient.device}")
            # use this line to copy the input tensor data.
            gradient = gradient.data.copy().view(xp.ndarray)

        if self.data.shape != gradient.shape:
            raise RuntimeError(f"grad can be implicitly created only for scalar outputs")


        if self.grad_fn:
            self.grad_fn.grad[self._output_idx] = gradient
            for grad_fn in toposort(self.grad_fn):
                gradient = grad_fn.apply(*grad_fn.grad)
                for i in range(len(grad_fn.next_functions)):
                    grad = gradient[i]
                    if grad is not None:
                        fn, ind = grad_fn.next_functions[i]
                        if fn.grad[ind] is None:
                            fn.grad[ind] = grad_fn.xp.ascontiguousarray(grad)
                        else:
                            fn.grad[ind] += grad
                grad_fn.clear()
        else:
            """
            import tortto as tt
            x=tt.tensor([1,2,3.], requires_grad=True)
            y=3*x
            y.backward(y)
            x.backward(x) # backprop from a tensor that requires_grad=True but grad_fn=None
            print(x.grad)
            """
            acc_grad = AccumulateGrad()
            acc_grad.variable = self
            acc_grad.grad=(gradient,)
            acc_grad.apply(*acc_grad.grad)








