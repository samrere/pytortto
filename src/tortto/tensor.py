import tortto
from tortto import np, cp, cp_ndarray, _int_zero
from .VariableFunctions import *
from .autograd.grad_fcn import *
from .autograd.grad_fcn import _slice, _view, _repeat, _expand, _cuda, _cpu
from .autograd.grad_ufunc import *

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
    def __init__(self, data, requires_grad=False, dtype=float32, copy=True):
        if data.__class__ is cp_ndarray:
            data = cp.array(data, dtype=dtype, copy=copy)
        else:
            data = np.array(data, dtype=dtype, copy=copy)

        self.data = data
        self.parents = []
        self.children = set()
        self.grad = _int_zero
        self.grad_fn = None
        self.grad_fn_param = None
        self.requires_grad = requires_grad

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

    def copy_(self, array):
        """
        1. the input can be tensor or array instead of tensor only as in pytorch
        2. used in module _load_from_state_dict. copy numpy array from checkpoint to tensor
           array from checkpoint is always numpy array.
        """
        if isinstance(array, Tensor):
            array = array.data
        if self.shape != array.shape:
            raise RuntimeError(f'The size of tensor a {self.shape} must match the size of tensor b {array.shape}')

        array = array.copy()  # copy array data
        from_class = array.__class__
        to_class = self.data.__class__
        if to_class is np.ndarray:
            if from_class is not np.ndarray:
                array = array.get()
        else:
            if from_class is np.ndarray:
                array = cp.array(array)
        self.data = array
        return self

    @property
    def T(self):
        return transpose(self)

    @property
    def device(self):
        if self.data.__class__ is cp_ndarray:
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
        return len(self.parents) == 0

    @property
    def ndim(self):
        return len(self.shape)

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

    @property
    def itemsize(self):
        return self.data.itemsize

    @property
    def strides(self):
        return self.data.strides

    @strides.setter
    def strides(self, value):
        self.data.strides = value

    def item(self):
        return self.data.item()

    def cuda(self):
        return _cuda(self)

    def cpu(self):
        return _cpu(self)

    def detach(self):
        # same as pytorch, where detached tensor share the same data with the original one.
        return Tensor(self.data, requires_grad=False, dtype=self.data.dtype, copy=False)

    def numpy(self):
        data = self.data
        if data.__class__ is cp_ndarray:
            raise TypeError(
                f"can't convert {self.device} device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return data

    def contiguous(self):
        """
        different from pytorch: tortto `contiguous` does it inplace.
        """
        xp = cp if self.data.__class__ is cp_ndarray else np
        self.data = xp.ascontiguousarray(self.data)
        return self

    def is_contiguous(self):
        return self.data.flags['C_CONTIGUOUS']

    def __eq__(self, other):
        if hasattr(self, 'data') and hasattr(other, 'data'):
            # use default dtype (float32) not bool, because any further operations on bool will result in float64:
            # tt.tensor([True, False], dtype=bool).mean() --> float64
            return tt.tensor(self.data == other.data, dtype=float32, copy=False)
        else:
            return False

    def __hash__(self):
        return id(self)

    def __repr__(self):
        device = f", device='{self.device}'" if self.device != 'cpu' else ''
        dtype = f', dtype={self.dtype}' if self.data is not None and self.dtype.type not in default_dtype else ''
        grad_fn = f', grad_fn=<{self.grad_fn.__name__}Backward>' if self.grad_fn else ''
        requires_grad = f', requires_grad=True' if self.requires_grad and not self.grad_fn else ''
        if self.data is None:
            s = str(None)
        else:
            s = np.array2string(self.data, separator=', ', precision=4).replace('\n', '\n' + ' ' * 7)
        return f'tensor({s}{device}{dtype}{grad_fn}{requires_grad})'

    def __add__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.add, self, other)

    __radd__ = __add__

    def __neg__(self):
        xp = cp if self.data.__class__ is cp_ndarray else np
        return compute_ufunc(xp.negative, self)

    def __sub__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.subtract, self, other)

    def __rsub__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.subtract, other, self)

    def __mul__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.multiply, self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.divide, self, other)

    def __rtruediv__(self, other):
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.divide, other, self)

    def __pow__(self, other):  # i.e. Tensor**3
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.power, self, other)

    def __rpow__(self, other):  # i.e. 3**Tensor
        xp = cp if self.data.__class__ is cp_ndarray else np
        if other.__class__ in Number:
            other = Tensor(xp.array(other), dtype=self.dtype)
        return compute_ufunc(xp.power, other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def sum(self, dim=None, keepdims=False):
        return sum(self, dim, keepdims)

    def mean(self, dim=None, keepdims=False):
        return mean(self, dim, keepdims)

    def var(self, dim=None, unbiased=True, keepdims=False):
        return var(self, dim, unbiased, keepdims)

    def __getitem__(self, key):
        # change indexing to slicing to keep dimension
        # if key.__class__ is int:
        #     key = slice(key, key + 1, None)
        # else:
        #     key = tuple([(slice(i,i+1,None) if i.__class__ is int else i) for i in key])
        return _slice(self, key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __array__(self, dtype=None):
        """
        useful when setting tensor values using another tensor:

        import tortto as tt
        x=tt.randn(2,4)
        x[:,::2]=tt.zeros((2,2))

        https://numpy.org/devdocs/user/basics.dispatch.html
        """
        return self.data.astype(dtype, copy=False)

    def __len__(self):
        return self.shape[0]

    def view(self, *newshape):
        if newshape[0].__class__ is not int:
            newshape = newshape[0]
        return _view(self, newshape)

    def flatten(self, start_dim=0, end_dim=-1):
        return flatten(self, start_dim, end_dim)

    def transpose(self, axes, axis2=None):
        if axis2 is not None:
            raise RuntimeError('tortto transpose is different from torch transpose, use swapaxes instead.')
        return transpose(self, axes)

    def swapaxes(self, axis1, axis2):
        return swapaxes(self, axis1, axis2)

    def moveaxis(self, source, destination):
        return moveaxis(self, source, destination)

    def chunk(self, chunks, dim=0):
        return chunk(self, chunks, dim=dim)

    def split(self, split_size_or_sections, dim=0):
        return split(self, split_size_or_sections, dim=dim)

    def repeat(self, *sizes):
        return _repeat(self, *sizes)

    def expand(self, *sizes):
        return _expand(self, *sizes)

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

    # def clone(self):
    #     if self._data is None:
    #         raise NotImplementedError('Not yet supporting slice/view clone')
    #     new = Tensor(None)
    #     new.__dict__ = self.__dict__.copy()
    #     new.data = xp.copy(self.data)
    #     new.parents = []
    #     new.children = set()
    #     return new

    def backward(self, gradient=None):
        if gradient is None:
            xp = cp if self.data.__class__ is cp_ndarray else np
            gradient = xp.expand_dims(xp.array(1, dtype=self.dtype), axis=tuple(range(self.ndim)))
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        if self.data.shape != gradient.shape:
            raise RuntimeError(
                f'Mismatch in shape: tensor has a shape of {self.shape} and gradient has a shape of {gradient.shape}')
        self.grad = gradient
        child_counts, parent_counts = count_children_and_parents(self)
        for node in toposort(self, child_counts):
            if node.requires_grad and node.grad_fn:
                #################### backward assertion
                if node.grad_fn is not _cuda and node.grad_fn is not _cpu:
                    assert node.grad.dtype == node.data.dtype, \
                        f'dtype assertion error during backward at {node.grad_fn.__name__}: ' \
                        f'grad is {node.grad.dtype} whereas node data is {node.data.dtype}'
                    assert node.grad.shape == node.data.shape, \
                        f'shape assertion error during backward at {node.grad_fn.__name__}: ' \
                        f'grad is {node.grad.shape} whereas node data is {node.data.shape}'
                    # assert array class: either both numpy arrays or both cupy arrays:
                    # only cupy array has attr "device", so they either both have .device, or both not have it.
                    # avoid using node.grad.__class__ because it can be array scalar( np.array(3)+np.array(4) = 7 ),
                    # so its class can be not ndarray
                    # numpy scalar: https://numpy.org/doc/stable/reference/arrays.scalars.html
                    assert hasattr(node.grad, 'device') == hasattr(node.data, 'device'), \
                        f'array class assertion error during backward at {node.grad_fn.__name__}: ' \
                        f'grad is {node.grad.__class__} whereas node data is {node.data.__class__}'
                ####################

                if node.parents:  # calc. gradient if node has parents
                    GRADIENTS_REGISTRY[node.grad_fn](node, node.grad, node.grad_fn_param)
            for child in node.children:
                parent_counts[child] -= 1
                assert parent_counts[child] >= 0, 'negative child counts'
                if parent_counts[child] == 0:
                    child.grad = _int_zero
                    child.parents = []
            node.children = set()
