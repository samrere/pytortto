from tortto import np, cp, cparray, cupy_is_loaded
from .function import *
from .helper import *

buildin_sum = sum

class Permute(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        yt0 = build_links(xp.transpose(xd0, params['dims']), grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dims=ctx.params['dims']
        xp = ctx.xp
        grad0=xp.transpose(gd0, axes=np.argsort(dims))
        return grad0


class Transpose(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        dim0, dim1 = params['dim0'], params['dim1']
        yt0 = build_links(xp.swapaxes(xd0, dim0,dim1), grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim0, dim1=ctx.params['dim0'],ctx.params['dim1']
        xp = ctx.xp
        grad0=xp.swapaxes(gd0, dim0, dim1)
        return grad0


class Mm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        if xd0.ndim!=2:
            raise RuntimeError("self must be a matrix")
        if xd1.ndim!=2:
            raise RuntimeError("mat2 must be a matrix")
        yt0 = build_links(xd0 @ xd1, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0=gd0 @ xd1.T
        if ctx.needs_input_grad[1]:
            grad1=xd0.T @ gd0
        return grad0, grad1

class Mv(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        if xd0.ndim!=2:
            raise RuntimeError("input must be a matrix")
        if xd1.ndim!=1:
            raise RuntimeError("vec must be a vector")
        yt0 = build_links(xd0 @ xd1, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        grad0, grad1 = None, None
        if ctx.needs_input_grad[0]:
            grad0=gd0[:,None] @ xd1[None]
        if ctx.needs_input_grad[1]:
            grad1=xd0.T @ gd0
        return grad0, grad1

class Bmm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        yt0 = build_links(xd0 @ xd1, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1 = ctx.saved_tensors
        grad0, grad1=None,None
        if ctx.needs_input_grad[0]:
            grad0= reverse_broadcast(gd0 @ xd1.swapaxes(-1, -2), xd0.shape)
        if ctx.needs_input_grad[1]:
            grad1= reverse_broadcast(xd0.swapaxes(-1, -2) @ gd0, xd1.shape)
        return grad0, grad1

class Addmm(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1, xt2 = inputs
        xd0, xd1, xd2 = xt0.data, xt1.data, xt2.data # input, mat1, mat2
        if xd1.ndim != 2:
            raise RuntimeError(f'mat1 must be a matrix, got {xd1.ndim}-D tensor')
        if xd2.ndim != 2:
            raise RuntimeError(f'mat2 must be a matrix, got {xd2.ndim}-D tensor')
        beta=params['beta']
        alpha=params['alpha']
        yd0=alpha*(xd1@xd2) if beta == 0 else beta*xd0+alpha*(xd1@xd2)
        yt0 = build_links(yd0, grad_fn=ctx)
        ctx.save_for_backward(xt0, xt1, xt2)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, xd1, xd2 = ctx.saved_tensors
        grad0, grad1, grad2 = None, None, None
        if ctx.needs_input_grad[0]:
            grad0= reverse_broadcast(gd0, xd0.shape)
        if ctx.needs_input_grad[1]:
            grad1=gd0 @ xd2.T
        if ctx.needs_input_grad[2]:
            grad2=xd1.T @ gd0
        return grad0, grad1, grad2

class Sum(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        dim=params['dim']
        keepdim=params['keepdim']
        yt0 = build_links(xp.sum(xd0, axis=dim, keepdims=keepdim), grad_fn=ctx)
        ctx.params['shape']=xd0.shape
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        if dim is None:
            grad0=xp.lib.stride_tricks.as_strided(gd0, shape=xd0_shape, strides=[0 for _ in xd0_shape])
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                gd0 = xp.expand_dims(gd0, dim)
            strides = list(gd0.strides)
            for i in dim:
                strides[i] = 0  # repeat along axis in x.shape
            grad0=xp.lib.stride_tricks.as_strided(gd0, shape=xd0_shape, strides=strides)
        return grad0



class Mean(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        dim=params['dim']
        keepdim=params['keepdim']
        yt0 = build_links(xp.mean(xd0, axis=dim, keepdims=keepdim), grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        dim = ctx.params['dim']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        if dim is None:
            grad0=xp.lib.stride_tricks.as_strided(xp.divide(gd0, xd0.size, dtype=gd0.dtype), shape=xd0.shape, strides=[0 for _ in xd0.shape])
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            if not keepdim:
                gd0 = xp.expand_dims(gd0, dim)
            N = 1
            strides = list(gd0.strides)
            for i in dim:
                N *= xd0.shape[i]
                strides[i] = 0  # repeat along axis in x.shape
            grad0=xp.lib.stride_tricks.as_strided(xp.divide(gd0, N, dtype=gd0.dtype), shape=xd0.shape, strides=strides)
        return grad0



class Var(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        dim=params['dim']
        unbiased=params['unbiased']
        keepdim=params['keepdim']
        yt0 = build_links(xp.var(xd0, axis=dim, ddof=unbiased, keepdims=keepdim), grad_fn=ctx)
        ctx.save_for_backward(xt0)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        dim = ctx.params['dim']
        unbiased = ctx.params['unbiased']
        keepdim = ctx.params['keepdim']
        xp = ctx.xp
        if not keepdim:
            gd0 = xp.expand_dims(gd0, dim)
        me = xp.mean(xd0, axis=dim, keepdims=True)
        if dim is None:  # if axis is None
            N = xd0.size  # all element
        else:
            if dim.__class__ is not tuple:
                dim = (dim,)
            N = 1
            for i in dim:
                N *= xd0.shape[i]
        grad0 = 2 * gd0 * xp.divide(xd0 - me, N - unbiased, dtype=gd0.dtype)
        return grad0



class Cat(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        dim = ctx.params['dim']
        xp = ctx.xp
        xdn=[]
        indices=[]
        for xt in inputs:
            xdn.append(xt.data)
            indices.append(xt.shape[dim])
        indices = np.cumsum(indices)
        yt0 = build_links(xp.concatenate(xdn, dim), grad_fn=ctx)
        ctx.params['indices'] = indices
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xp = ctx.xp
        dim=ctx.params['dim']
        indices = ctx.params['indices']
        gradn=xp.split(gd0, indices_or_sections=indices[:-1], axis=dim)
        return gradn



class Slice(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        yt0 = build_links(xd0[params['key']], grad_fn=ctx)
        ctx.params['shape']=xd0.shape
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape = ctx.params['shape']
        key = ctx.params['key']
        xp = ctx.xp
        grad0=xp.zeros(xd0_shape, dtype=gd0.dtype)
        grad0[key]=gd0
        return grad0


class View(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        yt0 = build_links(xd0.reshape(params['shape']), grad_fn=ctx)
        ctx.params['shape'] = xd0.shape
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0=gd0.reshape(ctx.params['shape'])
        return grad0

class Split(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim=params['dim']
        split_size_or_sections=params['split_size_or_sections']
        dim_size=xd0.shape[dim]
        if dim<0:
            dim+=xd0.ndim
        if split_size_or_sections.__class__ is int:
            split_size=split_size_or_sections
            ytn = tuple(
                build_links(
                    xd0[
                        tuple(
                            slice(None) if i != dim else slice(j, j + split_size) for i in range(xd0.ndim)
                        )
                    ],
                    grad_fn=ctx,
                    _output_idx=j // split_size
                )
                for j in range(0, dim_size, split_size)
            )
        else:
            sections=split_size_or_sections
            if buildin_sum(sections) != dim_size:
                raise RuntimeError(f"split_with_sizes expects split_sizes to sum exactly to 8 "
                                   f"(input tensor's size at dimension {dim}), but got split_sizes={sections}")
            sum_sections = np.cumsum(split_size_or_sections)
            ytn = tuple(
                build_links(
                    xd0[
                        tuple(
                            slice(None) if i != dim else slice(sum_sections[j] - sec, sum_sections[j]) for i in
                            range(xd0.ndim)
                        )
                    ],
                    grad_fn=ctx,
                    _output_idx=j
                )
                for j, sec in enumerate(sections)
            )
        ctx.save_for_backward(xt0)
        ctx.params['output_shapes'] = tuple(yt.shape for yt in ytn)
        return ytn
    @staticmethod
    def backward(ctx, *grad_outputs):
        xt0, = ctx.saved_tensors
        xp = ctx.xp
        output_shapes = ctx.params['output_shapes']
        dim = ctx.params['dim']
        grad0=xp.concatenate(
            [
                xp.zeros(output_shapes[i], dtype=xt0.dtype) if gdn is None else gdn
                for i, gdn in enumerate(grad_outputs)
            ],
            axis=dim
        )
        return grad0


class ToCopy(Function): # keep input _version: False
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        target_device = params['target_device']
        xp=ctx.xp
        if xp is cp:
            if target_device == 'cuda':
                return xt0
            else:
                yt0 = build_links(xd0.get(), grad_fn=ctx)
        else:
            if target_device == 'cpu':
                return xt0
            else:
                if not cupy_is_loaded:
                    raise RuntimeError("cupy not installed, can't use cuda")
                yt0 = build_links(cparray(xd0), grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        target_device = ctx.params['target_device']
        if target_device == 'cpu':
            grad0=cp.array(gd0)
        else:
            grad0=gd0.get()
        return grad0


class Repeat(Function): # keep input _version: False
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        sizes = params['sizes']
        xp = ctx.xp
        yt0 = build_links(xp.tile(xd0, sizes), grad_fn=ctx)
        ctx.params['shape']=xd0.shape
        ctx.params['yt0_strides']=yt0.strides
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0,=grad_outputs
        sizes = ctx.params['sizes']
        xd0_shape = ctx.params['shape']
        yt0_strides=ctx.params['yt0_strides']
        xd0_ndim=len(xd0_shape)
        leading_dims=tuple(range(len(sizes)))
        xp = ctx.xp
        target_shape=sizes+xd0_shape
        target_strides = yt0_strides[:-xd0_ndim]+\
                         tuple(xd0_shape[i]*yt0_strides[i-xd0_ndim] for i in range(xd0_ndim))+ \
                         yt0_strides[-xd0_ndim:]
        grad0=xp.lib.stride_tricks.as_strided(gd0, shape=target_shape, strides=target_strides).sum(leading_dims)
        return grad0


class Expand(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        sizes = params['sizes']
        xp = ctx.xp
        leading_dims = len(sizes) - len(xd0.shape)
        strides = [0] * leading_dims + list(xd0.strides)
        xd0_singleton_dims = [] # singleton axes to be summed during backward
        for i in range(len(sizes)):
            if i < leading_dims: # leading dimensions
                if sizes[i]<=0:
                    raise RuntimeError(f"The expanded size of the tensor ({sizes[i]}) isn't allowed in a leading, "
                                       f"non-existing dimension {i}")
            else:
                i-=len(sizes) # for non-leading dimensions, count backward
                if xd0.shape[i]==1:
                    if sizes[i]>1:
                        xd0_singleton_dims.append(i)
                        strides[i]=0
                else:
                    if sizes[i]!=-1 and xd0.shape[i]!=sizes[i]:
                        raise RuntimeError(f"The expanded size of the tensor ({sizes[i]}) must match the existing size "
                                           f"({xd0.shape[i]}) at non-singleton dimension {i+len(sizes)}.  "
                                           f"Target sizes: {sizes}.  Tensor sizes: {xd0.shape}")
        yd0 = xp.lib.stride_tricks.as_strided(xd0, shape=sizes, strides=strides) # a numpy/cupy array
        yt0 = build_links(yd0, grad_fn=ctx) # convert to nparray/cparray
        yt0.data._version = xd0._version  # keep version
        ctx.params = {'xd0_singleton_dims':xd0_singleton_dims, 'leading_dims':leading_dims}
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_singleton_dims = tuple(ctx.params['xd0_singleton_dims'])
        leading_dims = tuple(range(ctx.params['leading_dims']))
        grad0=gd0.sum(xd0_singleton_dims+leading_dims, keepdims=True).squeeze(leading_dims)
        return grad0

class Squeeze(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        if dim.__class__ is int:
            dim = (dim,)
        if dim is None:
            dim = tuple(range(xd0.ndim))
        squeeze_dims=tuple(i for i in dim if xd0.shape[i] == 1)
        if len(squeeze_dims) == 0:
            yt0 = build_links(xd0, grad_fn=ctx)
        else:
            xp = ctx.xp
            yt0 = build_links(xp.squeeze(xd0, squeeze_dims), grad_fn=ctx)
        params={'squeeze_dims':squeeze_dims}
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        squeeze_dims = ctx.params['squeeze_dims']
        xp = ctx.xp
        grad0 = xp.expand_dims(gd0, squeeze_dims)
        return grad0


class Unsqueeze(Function): # keep input _version: True
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        dim = params['dim']
        xp = ctx.xp
        yt0 = build_links(xp.expand_dims(xd0, dim), grad_fn=ctx)
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        dim = ctx.params['dim']
        xp = ctx.xp
        grad0 = xp.squeeze(gd0, dim)
        return grad0


class MaskedFill(Function): # keep input _version: False (except in-place)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data # input, val
        if xt1.ndim > 0:
            raise RuntimeError(f"masked_fill only supports a 0-dimensional value tensor, "
                               f"but got tensor with {xt1.ndim} dimension(s).")
        mask = params['mask']
        if mask.dtype.type is not np.bool_:
            raise RuntimeError(f"dtype of mask must be bool. "
                               f"Pass dtype=bool when constructing mask")
        flag=False
        if xd0.__class__ is cparray and xd1.__class__ is not cparray: # xd1 is a scaler, no need to convert it to cparray
            flag=True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            raise RuntimeError(f"masked_fill: Expected inputs to be on same device")

        key=(slice(None),)*(xd0.ndim-mask.ndim)+(mask.data,)
        if params['inplace']:
            inplace_precheck(xt0)
            xd0[key]=xd1
            yt0 = inplace_update(xt0, ctx)
        else:
            xd0=xd0.copy()
            xd0[key]=xt1.data
            yt0 = build_links(xd0, grad_fn=ctx)
        ctx.params['flag']=flag
        return yt0
    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        mask = ctx.params['mask']
        flag = ctx.params['flag']
        leading = (slice(None),) * (gd0.ndim - mask.ndim)
        grad0, grad1 = None, None
        if ctx.needs_input_grad[1]: # grad for value. Do this first because gd0 will be changed inplace next
            key = leading + (mask.data,)
            grad1=gd0[key].sum()
            if flag:
                grad1 = grad1.get()
        if ctx.needs_input_grad[0]: # grad for input
            key = leading + (mask.data,)
            grad0 = gd0
            grad0[key] = 0
        return grad0, grad1



class CopySlices(Function): # keep input _version: True (it's inplace)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        key = params['key']
        # convert xd1 to same array type as xd0
        flag = None
        if xd0.__class__ is cparray and xd1.__class__ is not cparray:
            xd1 = cp.array(xd1)
            flag = True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            xd1 = xd1.get()
            flag = False

        inplace_precheck(xt0)
        xd0[key] = xd1
        yt0 = xt0
        inplace_update(yt0, ctx)
        ctx.params['shapes']=(xd0.shape, xd1.shape)
        ctx.params['flag']=flag
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shapes']
        key = ctx.params['key']
        flag = ctx.params['flag']
        grad0, grad1 = None, None
        if ctx.needs_input_grad[1]: # grad for value. Do this first because gd0 will be changed inplace next
            grad1 = reverse_broadcast(gd0[key], xd1_shape)
            if flag is True:
                grad1 = grad1.get()
            elif flag is False:
                grad1 = cp.array(grad1)
        if ctx.needs_input_grad[0]:  # grad for input
            grad0 = gd0
            grad0[key] = 0
        return grad0, grad1

class Copy(Function): # keep input _version: True (it's inplace)
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data

        # convert xd1 to same array type as xd0
        flag = None
        if xd0.__class__ is cparray and xd1.__class__ is not cparray:
            xd1 = cp.array(xd1)
            flag = True
        elif xd0.__class__ is not cparray and xd1.__class__ is cparray:
            xd1 = xd1.get()
            flag = False

        inplace_precheck(xt0)
        xd0[...] = xd1
        yt0 = xt0
        inplace_update(yt0, ctx)
        ctx.params['flag'] = flag
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        grad0, grad1 = None, None
        flag = ctx.params['flag']
        if ctx.needs_input_grad[1]: # grad for value.
            grad1 = gd0
            if flag is True:
                grad1 = grad1.get()
            elif flag is False:
                grad1 = cp.array(grad1)
        if ctx.needs_input_grad[0]: # grad for input, zero
            grad0 = gd0
            grad0[...] = 0
        return grad0, grad1 # no grad for input

class Clamp(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, = inputs
        xd0 = xt0.data
        xp = ctx.xp
        if params['inplace']:
            inplace_precheck(xt0)
            if ctx.requires_grad:
                ctx.params['copy'] = xd0.copy()
            xp.clip(xd0, a_min=params['min'],a_max=params['max'],out=xd0)
            yt0 = inplace_update(xt0, ctx)
            ctx.save_for_backward(None)
        else:
            yt0 = build_links(xp.clip(xd0, a_min=params['min'],a_max=params['max']), grad_fn=ctx)
            ctx.save_for_backward(xt0)
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0, = ctx.saved_tensors
        if xd0 is None:
            xd0 = ctx.params['copy']
        lim_min=ctx.params['min']
        lim_max=ctx.params['max']
        if lim_min is not None:
            gd0[xd0<lim_min]=0
        if lim_max is not None:
            gd0[xd0>lim_max]=0
        return gd0
