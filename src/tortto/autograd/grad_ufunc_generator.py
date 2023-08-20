from pathlib import Path
import os
import yaml

to_save = ['']
Num_indents = [0]


def c(string):
    global to_save, Num_indents
    strings = string.split('\n')
    for string in strings:
        if string != '':
            to_save[0] += ' ' * Num_indents[0] * 4 + string + '\n'


def newline(num=1):
    global to_save
    to_save[0] += '\n' * num


def finished(name):
    global to_save
    to_save[0]=to_save[0].rstrip()+'\n'
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


def generate_grad_ufunc():
    # if os.path.exists(f'{Path(__file__).parent}/grad_ufunc.py'):
    #     return
    with open(rf'{Path(__file__).parent}/grad_ufunc_config.yaml') as file:
        yml = yaml.load(file, Loader=yaml.FullLoader)
        c('from tortto import np')
        c('from .function import *')
        c('from .helper import *')
        newline()
        c('"""')
        c('Auto-generated from grad_ufunc_generator.py')
        c('Any changes to this file will NOT be kept during next import')
        c('Instead, make changes to grad_ufunc_config.yaml to take effect')
        c('"""')
        newline(2)

        for name, config in yml.items():
            num_inputs = config['num_inputs']
            num_outputs = config['num_outputs']
            allow_inplace = config['allow_inplace']
            forward_inplace = config['forward_inplace']
            forward_outplace = config['forward_outplace']
            save_for_backward = config['save_for_backward']
            if save_for_backward is not None:
                save_for_backward_original=[i.strip() for i in save_for_backward.split(',')]
                len_saved = len(save_for_backward_original)
                save_for_backward_original=', '.join(save_for_backward_original)
            params = config['params']
            backward = config['backward']
            backward_additional=config['backward_additional']
            if backward_additional is None:
                backward_additional = ['' for _ in range(num_inputs)]
            assert len(backward)==num_inputs, 'bug'
            copy_xt0 = False
            grad0_not_req_xd0=False
            if allow_inplace and save_for_backward is not None:
                save_for_backward = [i.strip() for i in save_for_backward.split(',')]
                if 'xt0' in save_for_backward:
                    copy_xt0 = True
                    save_for_backward[0] = 'None'
                    # xt0 is only used in grad1, not grad0. As in Mul and Div
                    if len(backward)==2 and backward[0].find('xd0')==-1:
                        grad0_not_req_xd0=True
                save_for_backward = ', '.join(save_for_backward)


            # forward
            with indent(f'class {name}(Function):'):
                with indent('@staticmethod\ndef forward(ctx, *inputs, **params):'):
                    inputs = ', '.join(f'xt{i}' for i in range(num_inputs))
                    if num_inputs == 1:
                        inputs += ','
                    c(f"{inputs} = inputs")
                    c(f"{', '.join(f'xd{i}' for i in range(num_inputs))} = {', '.join(f'xt{i}.data' for i in range(num_inputs))}")
                    if forward_outplace.find('xp.') != -1:
                        c("xp = ctx.xp")
                    if allow_inplace:  # only single output if inplace
                        with indent(f"if params['inplace']:"):
                            c('inplace_precheck(xt0)')
                            if copy_xt0:
                                with indent('if xt1.requires_grad:' if grad0_not_req_xd0 else 'if ctx.requires_grad:'):
                                    c("ctx.params['copy'] = xd0.copy()")
                            c(f"{forward_inplace.replace('...', 'xd0').replace('///', 'xd1')}")
                            c('yt0 = inplace_update(xt0, ctx)')
                            if copy_xt0:
                                c(f"ctx.save_for_backward({save_for_backward})")
                        with indent('else:'):
                            c(f"yt0 = build_links({forward_outplace.replace('...', 'xd0').replace('///', 'xd1')}, grad_fn=ctx)")
                            if copy_xt0:
                                if grad0_not_req_xd0:
                                    with indent('if xt1.requires_grad:'):
                                        c(f"ctx.save_for_backward({save_for_backward_original})")
                                    with indent('else:'):
                                        c(f'ctx.save_for_backward({save_for_backward})')
                                else:
                                    c(f"ctx.save_for_backward({save_for_backward_original})")
                    else:
                        c(f"yt0 = build_links({forward_outplace.replace('...', 'xd0').replace('///', 'xd1')}, grad_fn=ctx)")
                    if save_for_backward is not None and copy_xt0 is False:
                        c(f"ctx.save_for_backward({save_for_backward})")
                    if params:
                        split=[f'xd{i}.{params}' for i in range(num_inputs)]
                        if len(split)==1:
                            c(f"ctx.params['{params}'] = {split[0]}")
                        else:
                            c(f"ctx.params['{params}'] = ({', '.join(split)})")
                    c(f"return {', '.join(f'yt{i}' for i in range(num_outputs))}")


                # backward
                newline()
                with indent('@staticmethod\ndef backward(ctx, *grad_outputs):'):
                    grads = ', '.join(f'gd{i}' for i in range(num_outputs))
                    if num_outputs == 1:
                        grads += ','
                    c(f"{grads} = grad_outputs")
                    if params:
                        c(f"{', '.join(f'x{i}_{params}' for i in range(num_inputs))} = ctx.params['{params}']")
                    if save_for_backward is not None:
                        save_for_backward_original = save_for_backward_original.replace('t', 'd')
                        if len_saved == 1:
                            save_for_backward_original += ','
                        save_for_backward = save_for_backward_original
                    if save_for_backward is not None:
                        c(f"{save_for_backward} = ctx.saved_tensors")
                    if copy_xt0 is True and not grad0_not_req_xd0:
                        with indent("if xd0 is None:"):
                            c("xd0 = ctx.params['copy']")
                    for b in backward:
                        if b.find('xp.') != -1:
                            c("xp = ctx.xp")

                    if num_inputs == 1:
                        assert len(backward) == 1, f'bug! number of input is 1, but number of gradient is {len(backward)}'
                        c(f"grad0 = {backward[0]}")
                        c(f'{backward_additional[0]}')
                    else:
                        c(', '.join([f'grad{i}' for i in range(num_inputs)])+' = '+', '.join([f'None' for i in range(num_inputs)]))
                        for i in range(num_inputs):
                            with indent(f'if ctx.needs_input_grad[{i}]:'):
                                if i==1 and grad0_not_req_xd0:
                                    with indent("if xd0 is None:"):
                                        c("xd0 = ctx.params['copy']")
                                c(f"grad{i} = {backward[i]}")
                                c(f'{backward_additional[i]}')
                    c(f"return {', '.join(f'grad{i}' for i in range(num_inputs))}")
            newline(2)

        finished('grad_ufunc.py')