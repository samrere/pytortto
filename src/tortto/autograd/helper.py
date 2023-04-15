import tortto as tt
from .grad_mode import is_grad_enabled
import tortto.autograd as au

def get_data(pair):
    if pair is None:
        return None
    tensor, version = pair
    if tensor._version == version:
        return tensor.data
    else:
        msg='' if tensor.grad_fn is None else f', which is the output of {tensor.grad_fn.__name__},'
        raise RuntimeError(f'one of the variables needed for gradient computation has been modified '
                           f'by an inplace operation: [shape: {tensor.shape}]{msg} is at version '
                           f'{tensor._version}; expected version {version} instead.')


def reverse_broadcast(result, target_shape: tuple):
    dim_diff = result.ndim - len(target_shape)
    if dim_diff < 0: # see matmul
        return result.reshape(target_shape)
    # sum all leading dimensions
    # except leading dimensions, sum all axis that are equal to 1, count from right
    dim0 = tuple(range(dim_diff))
    dim1 = tuple(i-len(target_shape) for i, value in enumerate(target_shape) if value == 1)
    result = result.sum(dim0+dim1, keepdims=True)
    result = result.squeeze(dim0)
    return result


# def count_children_and_parents(end_node):
#     """
#     link children during backward
#     """
#     parent_counts = dict()
#     child_counts = dict()
#     stack={end_node}
#     visited=set()
#     child_counts[end_node]=1 # put end_node into child_counts
#     while stack:
#         node=stack.pop()
#         visited.add(node)
#         parent_counts[node]=len(node.parents)
#         for p in node.parents:
#             if p is None: # ignore None parents (i.e. bias is False in Linear)
#                 parent_counts[node]-=1
#                 continue
#             p.children.add(node) # link children
#             if p not in child_counts:
#                 child_counts[p]=1
#             else:
#                 child_counts[p]+=1
#             if p not in visited:
#                 stack.add(p)
#     return child_counts, parent_counts
#
#
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
            for fn,_ in node.next_functions:
                if fn is None:
                    continue
                if fn.prev_function_counts == 0:
                    childless_nodes.append(fn)
                else:
                    fn.prev_function_counts -= 1
                    candidate.add(fn)

###################
## To be deleted ##
###################
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
