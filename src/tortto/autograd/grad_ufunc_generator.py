from pathlib import Path

import yaml

to_save = ['']
Num_indents = [0]


def c(string):
    global to_save, Num_indents
    strings = string.split('\n')
    for string in strings:
        to_save[0] += ' ' * Num_indents[0] * 4 + string + '\n'


def newline(num=1):
    global to_save
    to_save[0] += '\n' * num


def finished(name):
    global to_save
    ## save
    with open(f'{Path(__file__).parent}/{name}', 'w') as f:
        f.write(to_save[0])
    to_save[0] = ''


class indent:
    global Num_indents

    def __init__(self, string):
        c(string)

    def __enter__(self):
        Num_indents[0] += 1

    def __exit__(self, *args):
        Num_indents[0] -= 1


special = """class Mul(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if xt1.requires_grad:
                ctx.params = {'copy': xd0.copy()}
            yd0 = xp.multiply(xd0, xd1, out=xd0)
            yd0._version += 1
            yt0 = xt0
            yt0.requires_grad = requires_grad
            yt0.grad_fn = ctx
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = tt.tensor(xp.multiply(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params = {'shape':(xd0.shape, xd1.shape)}
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        if ctx.needs_input_grad[0]:
            xd1 = get_data(ctx.to_save[1])
            grad0 = reverse_broadcast(gd0 * xd1, xd0_shape)
        else:
            grad0 = None
        if ctx.needs_input_grad[1]:
            xd0 = ctx.params['copy'] if ctx.to_save[0] is None else get_data(ctx.to_save[0])
            grad1 = reverse_broadcast(gd0 * xd0, xd1_shape)
        else:
            grad1 = None
        return grad0, grad1


def mul(input, other):
    return Mul.apply(input, other, inplace=False)


multiply = mul


class Div(Function):
    @staticmethod
    def forward(ctx, *inputs, **params):
        xt0, xt1 = inputs
        xd0, xd1 = xt0.data, xt1.data
        xp = cp if xd0.__class__ is cparray else np
        requires_grad = xt0.requires_grad | xt1.requires_grad
        if params['inplace']:
            inplace_precheck(xt0)
            if xt1.requires_grad:
                ctx.params = {'copy': xd0.copy()}
            yd0 = xp.divide(xd0, xd1, out=xd0)
            yd0._version += 1
            yt0 = xt0
            yt0.requires_grad = requires_grad
            yt0.grad_fn = ctx
            ctx.save_for_backward(None, xt1)
        else:
            yt0 = tt.tensor(xp.divide(xd0, xd1), requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)
            ctx.save_for_backward(xt0, xt1)
        ctx.params = {'shape':(xd0.shape, xd1.shape)}
        return yt0

    @staticmethod
    def backward(ctx, *grad_outputs):
        gd0, = grad_outputs
        xd0_shape, xd1_shape = ctx.params['shape']
        xd1 = get_data(ctx.to_save[1])
        if ctx.needs_input_grad[0]:
            grad0 = reverse_broadcast(gd0 / xd1, xd0_shape)
        else:
            grad0 = None
        if ctx.needs_input_grad[1]:
            xd0 = ctx.params['copy'] if ctx.to_save[0] is None else get_data(ctx.to_save[0])
            grad1 = reverse_broadcast(-gd0 * xd0 / (xd1 * xd1), xd1_shape)
        else:
            grad1 = None
        return grad0, grad1


def div(input, other):
    return Div.apply(input, other, inplace=False)


divide = div
"""


def generate_grad_ufunc():
    with open(rf'{Path(__file__).parent}/grad_ufunc_config.yaml') as file:
        yml = yaml.load(file, Loader=yaml.FullLoader)
        c('from tortto import np, cp, cparray\nfrom .function import *\nfrom .helper import *')
        newline(2)
        c('"""')
        c("'x' is input\n'y' is output\n'g' is gradient")
        newline()
        c("'t' is for tensor\n'd' is for data (xparray)")
        newline()
        c('Use special formatting if the function allows inplace, but not all tensors in saved_tensors are used during backward.')
        c('Example: in Div, saved tensor xd0 (numerator) is not used during backward for numerator.')
        c("Therefore, if the denominator doesn't require grad, xd0 can be changed inplace and backward still works.")
        c("Same goes for Mul.")
        newline()
        c("import torch\nx=torch.tensor([1,2,3.], requires_grad=True)+0\ny=torch.tensor([4,5,6.], requires_grad=False)*1")
        c("z=x/y\nx+=1\nz.backward(torch.tensor([1,1,1]))")
        c('"""')
        newline(2)
        for name, config in yml.items():
            if config.get('special') is True:
                continue
            num_inputs = config['num_inputs']
            num_outputs = config['num_outputs']
            allow_inplace = config['allow_inplace']
            forward_inplace = config['forward_inplace']
            forward_outplace = config['forward_outplace']
            save_for_backward = config['save_for_backward']
            copy_xt0 = False
            if allow_inplace and save_for_backward is not None:
                save_for_backward_original = ', '.join([i.strip() for i in save_for_backward.split(',')])
                save_for_backward = [i.strip() for i in save_for_backward.split(',')]
                len_saved = len(save_for_backward)
                if 'xt0' in save_for_backward:
                    copy_xt0 = True
                    save_for_backward[0] = 'None'
                save_for_backward = ', '.join(save_for_backward)
            params = config['params']
            backward = config['backward']
            if len(backward) == 1:
                backward = backward * num_inputs
            with indent(f'class {name}(Function):'):
                with indent('@staticmethod\ndef forward(ctx, *inputs, **params):'):
                    inputs = ', '.join(f'xt{i}' for i in range(num_inputs))
                    if num_inputs == 1:
                        inputs += ','
                    c(f"{inputs} = inputs")
                    c(f"{', '.join(f'xd{i}' for i in range(num_inputs))} = {', '.join(f'xt{i}.data' for i in range(num_inputs))}")
                    if forward_outplace.find('xp.') != -1:
                        c("xp = cp if xd0.__class__ is cparray else np")
                    c(f"requires_grad = {' | '.join(f'xt{i}.requires_grad' for i in range(num_inputs))}")
                    if allow_inplace:  # only single output if inplace
                        with indent(f"if params['inplace']:"):
                            c('inplace_precheck(xt0)')
                            if copy_xt0:
                                c("ctx.params = {'copy': xd0.copy()}")
                            c(f"yd0 = {forward_inplace.replace('...', 'xd0').replace('///', 'xd1')}")
                            c("yd0._version += 1\nyt0 = xt0\nyt0.requires_grad = requires_grad\nyt0.grad_fn = ctx")
                            if copy_xt0:
                                c(f"ctx.save_for_backward({save_for_backward})")
                        with indent('else:'):
                            c(f"yt0 = tt.tensor({forward_outplace.replace('...', 'xd0').replace('///', 'xd1')}, requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)")
                            if copy_xt0:
                                c(f"ctx.save_for_backward({save_for_backward_original})")
                    else:
                        c(f"yt0 = tt.tensor({forward_outplace.replace('...', 'xd0').replace('///', 'xd1')}, requires_grad=requires_grad, copy=False, _output_idx=0, grad_fn=ctx)")
                    if save_for_backward is not None and copy_xt0 is False:
                        c(f"ctx.save_for_backward({save_for_backward})")
                    if params:
                        c(f"ctx.params = {{'{params}':({', '.join(f'xd{i}.{params}' for i in range(num_inputs))})}}")
                    c(f"return {', '.join(f'yt{i}' for i in range(num_outputs))}")
                newline()
                with indent('@staticmethod\ndef backward(ctx, *grad_outputs):'):
                    grads = ', '.join(f'gd{i}' for i in range(num_outputs))
                    if num_outputs == 1:
                        grads += ','
                    c(f"{grads} = grad_outputs")
                    if params:
                        c(f"{', '.join(f'xd{i}_{params}' for i in range(num_inputs))} = ctx.params['{params}']")
                    if save_for_backward is not None:
                        save_for_backward_original = save_for_backward_original.replace('t', 'd')
                        if len_saved == 1:
                            save_for_backward_original += ','
                        save_for_backward = save_for_backward_original
                    if save_for_backward is not None:
                        c(f"{save_for_backward} = ctx.saved_tensors")
                    if copy_xt0 is True:
                        with indent("if xd0 is None:"):
                            c("xd0 = ctx.params['copy']")
                    infer = True
                    for b in backward:
                        if b.find('xp.') != -1:
                            c("xp = cp if gd0.__class__ is cparray else np")
                        if b.find('///') != -1:
                            infer = False
                    if num_inputs == 1:
                        assert len(backward) == 1, 'bug!'
                        c(f"grad0 = {backward[0].replace('...', '0')}")
                    else:
                        if infer:
                            for i in range(num_inputs):
                                c(f"grad{i} = {backward[i].replace('...', str(i))} if ctx.needs_input_grad[{i}] else None")
                        else:
                            for i in range(num_inputs):
                                c(f"grad{i} = {backward[i].replace('...', '0').replace('///', '1')} if ctx.needs_input_grad[{i}] else None")
                    c(f"return {', '.join(f'grad{i}' for i in range(num_inputs))}")
            if config['alias']:
                newline(2)
                if num_inputs == 1:
                    with indent(f"def {name.lower()}(input):"):
                        c(f"return {name}.apply(input{', inplace=False' if allow_inplace else ''})")
                elif num_inputs == 2:
                    with indent(f"def {name.lower()}(input, other):"):
                        c(f"return {name}.apply(input, other{', inplace=False' if allow_inplace else ''})")
            if config['long']:
                newline(2)
                c(f"{config['long']} = {name.lower()}")
            if config['alias_']:
                newline(2)
                if num_inputs == 1:
                    with indent(f"def {name.lower()}_(input):"):
                        c(f"return {name}.apply(input{', inplace=True' if allow_inplace else ''})")
                elif num_inputs == 2:
                    with indent(f"def {name.lower()}_(input, other):"):
                        c(f"return {name}.apply(input, other{', inplace=True' if allow_inplace else ''})")
            if config['long_']:
                newline(2)
                c(f"{config['long_']} = {name.lower()}")
            newline(2)

        to_save[0] += special
        newline()

        finished('grad_ufunc.py')
