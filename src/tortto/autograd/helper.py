import tortto as tt
from .grad_mode import is_grad_enabled


def get_data(pair):
    tensor_xparray, version = pair

    if tensor_xparray._version == version:
        if tensor_xparray.__class__ is tt.Tensor:
            return tensor_xparray.data
        else:
            return tensor_xparray
    else:
        msg='' if tensor_xparray.grad_fn is None else f', which is the output of {tensor_xparray.grad_fn.__name__},'
        raise RuntimeError(f'one of the variables needed for gradient computation has been modified '
                           f'by an inplace operation: [shape: {tensor_xparray.shape}]{msg} is at version '
                           f'{tensor_xparray._version}; expected version {version} instead.')


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
