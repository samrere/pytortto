from tortto import *

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True, **kwargs):
        dtype = float32
        if isinstance(data, Tensor):
            data = data.data
            dtype = data.dtype
        else:
            if data is not None:
                raise TypeError(f'input must be Tensor, not {data.__class__.__name__}')
        super().__init__(data, requires_grad=requires_grad, dtype=dtype, copy=False) # no copy, same as in pytorch

    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()
