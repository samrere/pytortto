import tortto as tt
from .grad_mode import *


def inplace_precheck(xt):
    if is_grad_enabled() and xt.requires_grad and xt.grad_fn is None:
        raise RuntimeError(f"a leaf Variable that requires grad is being used in an in-place operation.")


def inplace_update(tensor, grad_fn):
    tensor.data._version[0] += 1
    tensor._output_idx = 0
    tensor.requires_grad = grad_fn.requires_grad
    if grad_fn.requires_grad:
        tensor.grad_fn = grad_fn
    return tensor


def get_data(pair):
    if pair is None:
        return None
    tensor, version = pair  # version is saved as an int, not a list
    if tensor._version == version:
        return tensor.data
    else:
        msg = '' if tensor.grad_fn is None else f', which is the output of {tensor.grad_fn.__class__.__name__},'
        raise RuntimeError(f'one of the variables needed for gradient computation has been modified '
                           f'by an inplace operation: [shape: {tensor.shape}]{msg} is at version '
                           f'{tensor._version}; expected version {version} instead.')


def reverse_broadcast(result, target_shape: tuple):
    dim_diff = result.ndim - len(target_shape)
    if dim_diff < 0:  # see matmul
        return result.reshape(target_shape)
    # sum all leading dimensions
    # except leading dimensions, sum all axis that are equal to 1, count from right
    dim0 = tuple(range(dim_diff))
    dim1 = tuple(i - len(target_shape) for i, value in enumerate(target_shape) if value == 1)
    result = result.sum(dim0 + dim1, keepdims=True)
    result = result.squeeze(dim0)
    return result


def build_links(data, grad_fn, copy=False, _output_idx=0):
    requires_grad = grad_fn.requires_grad & is_grad_enabled()
    if requires_grad:
        return tt.tensor(data, requires_grad=True, grad_fn=grad_fn, copy=copy, _output_idx=_output_idx)
    else:
        return tt.tensor(data, copy=copy)


def toposort(end_node):
    # yield childless node
    childless_nodes = [end_node]
    """
        x-->y
        |
        |
        z
    if backward from `y`, `x` will be next_functions of `y`, but its prev_function_counts is 2 (`y` and `z`), 
    after subtracting 1 from its `prev_function_counts`, node `y` won't be able to be put into `childless_nodes`.
    we use a set called the `candidate` to store such nodes.
    
    Nodes will be firstly popped from the `childless_nodes` list.
    Candidate will only be popped if `childless_nodes` is empty.
    """
    candidate = set()
    while childless_nodes or candidate:
        node = childless_nodes.pop() if childless_nodes else candidate.pop()
        yield node
        if node.next_functions is not None:
            for fn, _ in node.next_functions:
                if fn is None:
                    continue
                fn.prev_function_counts -= 1
                if fn.prev_function_counts == 0:
                    childless_nodes.append(fn)
                    if fn in candidate:
                        candidate.remove(fn)
                else:
                    candidate.add(fn)
