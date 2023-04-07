import tortto as tt

from .grad_mode import *

import tortto.autograd as au

GRADIENTS_REGISTRY = dict()


def compute_ufunc(ufunc, *inputs, params=None, **kwargs):
    if params is None:
        params=dict()
    scalars = []
    requires_grad = False
    for input in inputs:
        if input.__class__ is tt.Tensor:
            scalars.append(input.data)
            if input.requires_grad:
                requires_grad = True
        else:
            raise NotImplementedError(
                f'bug at {ufunc.__name__}: input should be Tensor, not {input.__class__.__name__}')
    value = ufunc(*scalars, **kwargs)
    output = build_links(value, requires_grad, ufunc, *inputs, **params)
    return output


def build_links(value, requires_grad, op, *inputs, **params):
    #################### forward assertion
    if op is not au.grad_fcn._cuda and op is not au.grad_fcn._cpu:
        for i in inputs:
            if i is not None:
                assert value.dtype == i.dtype, \
                    f'dtype assertion error during forward at {op.__name__}: ' \
                    f'value is {value.dtype} whereas input is {i.dtype}'
                assert hasattr(value.data, 'device') == hasattr(i.data, 'device'), \
                    f'array class assertion error during forward at {op.__name__}: ' \
                    f'value is {value.data.__class__} whereas input is {i.data.__class__}'
    ####################

    if not is_grad_enabled():
        requires_grad = False

    inplace = value is inputs[0].data
    if inplace:
        value._version+=1

    output = tt.tensor(value, requires_grad=requires_grad, copy=False)

    # early exit if not require grad
    if not requires_grad:
        return output

    # Nones are recorded and can be duplicated (i.e. when weight and bias are both False in batch_norm)
    output.parents.extend([None if i is None else (i,i._version) for i in inputs])
    output.output_version = output._version
    output.grad_fn = op
    output.grad_fn_param = params
    return output


def inplace_precheck(fn):
    def wrapper(*args):
        if is_grad_enabled():
            for arg in args:
                if arg.__class__ is tt.Tensor and arg.requires_grad and arg.grad_fn is None:
                    raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        return fn(*args)

    return wrapper


def get_data(tuple_or_tensor):
    if tuple_or_tensor.__class__ is tuple:
        tensor, version = tuple_or_tensor
    else:
        tensor, version=tuple_or_tensor,tuple_or_tensor.output_version

    if tensor._version == version:
        return tensor.data
    else:
        raise RuntimeError(f'one of the variables needed for gradient computation has been modified '
                           f'by an inplace operation: [shape: {tensor.shape}], which is the output of '
                           f'{tensor.grad_fn.__name__}, is at version {tensor._version}; '
                           f'expected version {version} instead.')


def register_gradients(*gradients):
    def wrapper(fn):
        for gradient in gradients:
            if gradient in GRADIENTS_REGISTRY:  # precheck
                raise ValueError(f'{gradient.__name__} already in registry')
        for gradient in gradients:  # gradients from ufunc may be same
            GRADIENTS_REGISTRY[gradient] = fn
        return fn

    return wrapper


def reverse_broadcast(result, target_shape: tuple):
    # sum all leading dimensions
    # except leading dimensions, sum all axis that are equal to 1, count from right
    axis0 = tuple(range(result.ndim - len(target_shape)))
    axis1 = tuple(i - len(target_shape) for i, value in enumerate(target_shape) if value == 1)
    result = result.sum(axis=axis0 + axis1, keepdims=True)
    result = result.squeeze(axis=axis0)
    return result


def count_children_and_parents(end_node):
    """
    link children during backward
    """
    parent_counts = dict()
    child_counts = dict()
    stack = {end_node}
    visited = set()
    child_counts[end_node] = 1  # put end_node into child_counts
    while stack:
        node = stack.pop()
        visited.add(node)
        parent_counts[node] = len(node.parents)
        for pair in node.parents:
            if pair is None:  # ignore None parents (i.e. bias is False in Linear)
                parent_counts[node] -= 1
                continue
            p = pair[0]
            p.children.add(node)  # link children
            if p not in child_counts:
                child_counts[p] = 1
            else:
                child_counts[p] += 1
            if p not in visited:
                stack.add(p)
    return child_counts, parent_counts


def toposort(end_node, child_counts):
    # yield childless node
    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for pair in node.parents:
            if pair is None:
                continue
            p=pair[0]
            if child_counts[p] == 1:
                childless_nodes.append(p)
            else:
                child_counts[p] -= 1
