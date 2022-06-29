import tortto as tt

from .grad_mode import *

import tortto.autograd as au

GRADIENTS_REGISTRY = dict()

def compute_ufunc(ufunc, *inputs, **kwargs):
    scalars = []
    requires_grad = False
    for input in inputs:
        if isinstance(input, tt.Tensor):
            scalars.append(input.data)
            if input.requires_grad:
                requires_grad = True
        else:
            raise NotImplementedError(f'bug at {ufunc.__name__}: input should be Tensor, not {input.__class__.__name__}')
    value = ufunc(*scalars, **kwargs)
    output = build_links(value, requires_grad, ufunc, *inputs)
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

    output = tt.tensor(value, requires_grad=requires_grad, copy=False)

    # early exit if not require grad
    if not requires_grad:
        return output

    # Nones are recorded and can be duplicated (i.e. when weight and bias are both False in batch_norm)
    output.parents.extend(inputs)
    output.grad_fn = op
    output.grad_fn_param = params
    return output


def register_gradients(*gradients):
    def wrapper(fn):
        for gradient in gradients:
            if gradient in GRADIENTS_REGISTRY: # precheck
                raise ValueError(f'{gradient.__name__} already in registry')
        for gradient in gradients: # gradients from ufunc may be same
            GRADIENTS_REGISTRY[gradient] = fn
        return fn
    return wrapper


def reverse_broadcast(result, target_shape: tuple):
    # sum all leading dimensions
    # except leading dimensions, sum all axis that are equal to 1, count from right
    axis0 = tuple(range(result.ndim - len(target_shape)))
    axis1 = tuple(i-len(target_shape) for i, value in enumerate(target_shape) if value == 1)
    result = result.sum(axis=axis0+axis1, keepdims=True)
    result = result.squeeze(axis=axis0)
    return result


def count_children_and_parents(end_node):
    """
    link children during backward
    """
    parent_counts = dict()
    child_counts = dict()
    stack={end_node}
    visited=set()
    child_counts[end_node]=1 # put end_node into child_counts
    while stack:
        node=stack.pop()
        visited.add(node)
        parent_counts[node]=len(node.parents)
        for p in node.parents:
            if p is None: # ignore None parents (i.e. bias is False in Linear)
                parent_counts[node]-=1
                continue
            p.children.add(node) # link children
            if p not in child_counts:
                child_counts[p]=1
            else:
                child_counts[p]+=1
            if p not in visited:
                stack.add(p)
    return child_counts, parent_counts



def toposort(end_node,child_counts):
    # yield childless node
    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for p in node.parents:
            if p is None:
                continue
            if child_counts[p] == 1:
                childless_nodes.append(p)
            else:
                child_counts[p] -= 1
