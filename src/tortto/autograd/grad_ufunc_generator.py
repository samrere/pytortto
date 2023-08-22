from pathlib import Path
import re
import yaml

to_save = ['']
Num_indents = [0]


def c(string):
    global to_save, Num_indents
    strings = string.split('\n')
    for string in strings:
        string=string.rstrip()
        string=string.replace('>>> ','')
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

def get(adict, key, otherwise=None):
    y=adict.get(key)
    if y is None:
        y=otherwise
    return y
def remove_params_from_string(s):
    """
    "fcn(xd0, a_min=params['min'], a_max=ctx.params['max'], out=xd0)" -> "fcn(xd0, a_min=min, a_max=max, out=xd0)"
    """
    return re.sub(r'[ctx.]*params\[[\'\"]?(.*?)[\'\"]?]', r'\1',s)
def generate_grad_ufunc():
    # import os
    # if os.path.exists(f'{Path(__file__).parent}/grad_ufunc.py'):
    #     return
    def precheck():
        if allow_inplace:
            assert num_outputs == 1, f'{name}: allow inplace is True, but more than 1 num_outputs'

        assert len(backward) == num_inputs, f'{name}: backward length should be equal to num_inputs'

        joined_string_fwd = ' '.join(forward_inplace + forward_outplace)
        xd = re.findall(r'xd[0-9]', joined_string_fwd)
        assert len(xd) == 0, f"{name}: {xd} exist in forward"

        joined_string_bwd = ' '.join([backward_common] + backward)
        gd = re.findall(r'gd[0-9]', joined_string_bwd)
        assert len(gd) == 0, f"{name}: {gd} exist in backward"

        return joined_string_fwd, joined_string_bwd

    with open(rf'{Path(__file__).parent}/grad_ufunc_config.yaml') as file:
        yml = yaml.load(file, Loader=yaml.FullLoader)
        c('from .function import *')
        c('from .helper import *')
        c('from tortto import cp, cparray, cupy_is_loaded')
        newline()
        c('"""')
        c('Auto-generated from grad_ufunc_generator.py')
        c('Any changes to this file will NOT be kept during next import')
        c('Instead, make changes to grad_ufunc_config.yaml to take effect')
        c('"""')
        newline(2)

        for name, config in yml.items():
            comment = get(config, 'comment','')
            num_inputs = config['num_inputs']
            num_outputs = config['num_outputs']
            allow_inplace = config['allow_inplace']
            forward_inplace = get(config, 'forward_inplace',[''])
            forward_outplace = get(config, 'forward_outplace',[''])
            save_for_backward = config['save_for_backward']
            backward_common = get(config, 'backward_common','')
            if save_for_backward is not None:
                save_for_backward_original=[i.strip() for i in save_for_backward.split(',')]
                len_saved = len(save_for_backward_original)
                save_for_backward_original=', '.join(save_for_backward_original)
            save_params = get(config, 'save_params','')
            if save_params:
                save_params = [_.strip() for _ in save_params.split()]
            save_class_params = config['save_class_params']
            backward = config['backward']
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

            joined_string_fwd, joined_string_bwd = precheck()
            used_gd = re.findall(r'\$([0-9])', joined_string_bwd)
            used_gd = set(int(s) for s in used_gd)


            # forward
            with indent(f'class {name}(Function):'):
                c(comment)
                with indent('@staticmethod\ndef forward(ctx, *inputs, **params):'):
                    inputs = ', '.join(f'xt{_}' for _ in range(num_inputs))
                    if num_inputs == 1:
                        inputs += ','
                    c(f"{inputs} = inputs")
                    c(f"{', '.join(f'xd{_}' for _ in range(num_inputs))} = {', '.join(f'xt{_}.data' for _ in range(num_inputs))}")
                    all_required_params = re.findall(r'[ctx.]?params\[[\'\"]?(.*?)[\'\"]?]', joined_string_fwd)
                    all_required_params = set(all_required_params)
                    for p in all_required_params:
                        c(f"{p} = params['{p}']")
                    forward_inplace = [remove_params_from_string(s) for s in forward_inplace]
                    forward_outplace = [remove_params_from_string(s) for s in forward_outplace]

                    if joined_string_fwd.find('xp.') != -1 or joined_string_fwd.find('xp ') != -1:
                        c("xp = ctx.xp")

                    if allow_inplace:  # only single output if inplace
                        with indent(f"if params['inplace']:"):
                            c('inplace_precheck(xt0)')
                            if copy_xt0:
                                with indent('if xt1.requires_grad:' if grad0_not_req_xd0 else 'if ctx.requires_grad:'):
                                    c("ctx.params['copy'] = xd0.copy()")
                            c(f"{forward_inplace[0].replace('$', 'xd')}")
                            c('yt0 = inplace_update(xt0, ctx)')
                            if copy_xt0:
                                c(f"ctx.save_for_backward({save_for_backward})")
                        with indent('else:'):
                            c(f"{forward_outplace[0].replace('$', 'xd').replace('@','yd0')}")
                            c(f"yt0 = build_links(yd0, grad_fn=ctx)")
                            if copy_xt0:
                                if grad0_not_req_xd0:
                                    with indent('if xt1.requires_grad:'):
                                        c(f"ctx.save_for_backward({save_for_backward_original})")
                                    with indent('else:'):
                                        c(f'ctx.save_for_backward({save_for_backward})')
                                else:
                                    c(f"ctx.save_for_backward({save_for_backward_original})")
                    else:
                        for i in range(num_outputs):
                            c(forward_outplace[i].replace('$','xd').replace('@',f'yd{i}'))
                            if i in used_gd:
                                c(f"yt{i} = build_links(yd{i}, grad_fn=ctx{'' if i==0 else f', _output_idx={i}'})")
                            else:
                                c(f"yt{i} = tt.tensor(yd{i}, copy=False)")
                    if save_for_backward is not None and copy_xt0 is False:
                        c(f"ctx.save_for_backward({save_for_backward})")
                    for s in save_params:
                        c(f"ctx.params['{s}'] = {s}")
                    if save_class_params:
                        split=[f'xd{_}.{save_class_params}' for _ in range(num_inputs)]
                        if len(split)==1:
                            c(f"ctx.params['{save_class_params}'] = {split[0]}")
                        else:
                            c(f"ctx.params['{save_class_params}'] = ({', '.join(split)})")
                    c(f"return {', '.join(f'yt{_}' for _ in range(num_outputs))}")


                # backward
                newline()
                with indent('@staticmethod\ndef backward(ctx, *grad_outputs):'):
                    grads = ', '.join(f'gd{i}' if i in used_gd else '_' for i in range(num_outputs))
                    if num_outputs == 1:
                        grads += ','
                    c(f"{grads} = grad_outputs")

                    if save_for_backward is not None:
                        save_for_backward_original = save_for_backward_original.replace('t', 'd')
                        if len_saved == 1:
                            save_for_backward_original += ','
                        save_for_backward = save_for_backward_original
                    if save_for_backward is not None:
                        c(f"{save_for_backward} = ctx.saved_tensors")

                    for sp in save_params:
                        c(f"{sp} = ctx.params['{sp}']")
                    if save_class_params:
                        c(f"{', '.join(f'xd{i}_{save_class_params}' for i in range(num_inputs))} = ctx.params['{save_class_params}']")

                    all_required_params = re.findall(r'[ctx.]?params\[[\'\"]?(.*?)[\'\"]?]', joined_string_bwd)
                    all_required_params = set(all_required_params)
                    for sp in save_params:
                        all_required_params.discard(sp)
                    for p in all_required_params:
                        c(f"{p} = ctx.params['{p}']")
                    backward_common = remove_params_from_string(backward_common)
                    backward = [remove_params_from_string(s) for s in backward]

                    if copy_xt0 is True and not grad0_not_req_xd0:
                        with indent("if xd0 is None:"):
                            c("xd0 = ctx.params['copy']")

                    flag=False
                    if backward_common is not None and backward_common.find('xp.') != -1:
                        flag=True
                    for b in backward:
                        if b.find('xp.') != -1:
                            flag=True
                    if flag:
                        c("xp = ctx.xp")

                    c(backward_common)

                    if num_inputs == 1:
                        assert len(backward) == 1, f'bug! number of input is 1, but number of gradient is {len(backward)}'
                        c(backward[0].replace('$', 'gd').replace('@', 'grad0'))
                    else:
                        c(', '.join([f'grad{i}' for i in range(num_inputs)])+' = '+', '.join([f'None' for i in range(num_inputs)]))
                        for i in range(num_inputs):
                            with indent(f'if ctx.needs_input_grad[{i}]:'):
                                if i==1 and grad0_not_req_xd0:
                                    with indent("if xd0 is None:"):
                                        c("xd0 = ctx.params['copy']")
                                c(backward[i].replace('$', 'gd').replace('@',f'grad{i}'))
                    c(f"return {', '.join(f'grad{i}' for i in range(num_inputs))}")
            newline(2)
        finished('grad_ufunc.py')