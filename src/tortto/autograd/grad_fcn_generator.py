import re
from pathlib import Path

import numpy as np
import yaml

to_save = ['']
Num_indents = [0]


def get(adict, key, default):
    y = adict.get(key)
    return default if y is None else y


def c(string):
    global to_save, Num_indents
    strings = string.split('\n')
    for string in strings:
        string = string.rstrip()
        if string != '':
            to_save[0] += ' ' * Num_indents[0] * 4 + string + '\n'


def newline(num=1):
    global to_save
    to_save[0] += '\n' * num


def finished(name):
    global to_save
    to_save[0] = to_save[0].rstrip() + '\n'
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


def find_word(target, string):
    return re.search(rf'(?<!\w)((?<!\.){target}(?!_))(?!\w)', string)


def insert(string, target, add):
    if len(target) == 1:
        return re.sub(rf'(?<!\w){target}(\d)(?!\w)', rf'{target}{add}\1', string)
    else:
        return re.sub(rf'(?<!\w)([{target}])(\d)(?!\w)', rf'\1{add}\2', string)


class Config:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.params = self.parse_params()
        self.backward_combined, self.backward_common, self.gradient, self.gradient_order = self.parse_backward()
        self.forward_combined, self.gradient_combined, self.forward_common, self.allow_inplace, \
        self.inplace, self.outplace, self.inputs, self.outputs, \
        self.saved_tensors, self.saved_arrays, self.saved_property = self.parse_forward()

    def parse_params(self):
        params = get(self.config, 'params', '')
        if params:
            return sorted(set(i.strip() for i in params.split(',')))
        return []

    def write_comment_to_text(self):
        loaded = get(self.config, 'comment', '').strip()
        if loaded:
            c('\n'.join('# ' + s for s in loaded.split('\n')))

    def parse_backward(self):
        backward_common = get(self.config['backward'], 'common', '').strip()

        gradient_load = get(self.config['backward'], 'gradient', dict())
        gradient_order = []
        if type(gradient_load) is dict:
            gradient = [''] * len(gradient_load)
            for id, code in gradient_load.items():
                order = set(re.findall(r'(?<=\n)grad(\d)(?=\s*?=)', '\n' + code))
                if len(order) != 1:
                    raise RuntimeError(f'{self.name}: number of gradient output at {id} should be 1')
                gradient_order.append(int(list(order)[0]))
                gradient[id] = code
        else:  # it's string
            gradient = [gradient_load.strip()]

        backward_combined = '\n' + ('\n'.join([backward_common] + gradient) if backward_common else '\n'.join(gradient))

        return backward_combined, backward_common, gradient, gradient_order

    def parse_forward(self):
        forward_common = get(self.config['forward'], 'common', '')
        inplace_load = get(self.config['forward'], 'inplace', dict())
        if type(inplace_load) is dict:
            inplace = [''] * len(inplace_load)
            for id, code in inplace_load.items():
                inplace[id] = re.sub(r'(?<!\w)y(?!\w)', f'y{id}', code.strip())  # replace y with y0, y1 etc
        else:
            inplace = [inplace_load.strip()]

        allow_inplace = len(inplace) != 0

        outplace_load = get(self.config['forward'], 'outplace', dict())
        if type(outplace_load) is dict:
            outplace = [''] * len(outplace_load)
            for id, code in outplace_load.items():
                outplace[id] = re.sub(r'(?<!\w)y(?!\w)', f'y{id}', code.strip())
        else:
            outplace = [outplace_load.strip()]

        forward_combined = '\n' + forward_common + '\n'.join(inplace + outplace)
        inputs = re.findall(r'(?<!\w)(x\d)(?!\w)', forward_combined)  # find inputs such as x0, x1
        inputs = set(inputs)
        outputs = set(f'y{i}' for i in range(max(len(inplace), len(outplace))))

        def search_tensorarray(aset, string):
            # search tensors or arrays in string, such as x0, including transpose x0.T and class method x0.transpose()
            # excluding tensor property like x0.shape
            clean = {i for i in aset if re.search(rf'(?<!\w)({i}(?!\.))(?!\w)', string)}  # such as x0
            T = {i for i in aset if re.search(rf'(?<!\w){i}\.T(?!\w)', string)}  # such as x0.T
            T = {i.split('.')[0] for i in T}
            class_methods = {i for i in aset if
                             re.search(rf'(?<!\w){i}\.[a-z]+?\((?!\w)', string)}  # such as x0.transpose()
            class_methods = {i.split('.')[0] for i in class_methods}
            return clean | T | class_methods

        # get saved tensor, tensor property and interm_result
        all_tensors = inputs | outputs
        saved_tensors = []
        redefined = set(re.findall(r'(?<=\n)(\w*?)(?=\s*?=)',
                                   '\n' + self.backward_common))  # tensors or arrays redefined in backward common
        for _ in range(len(self.gradient)):
            saved_tensors.append(search_tensorarray(all_tensors, self.gradient[_]) - redefined)
        saved_tensors.append(
            search_tensorarray(all_tensors, self.backward_common) - redefined)  # same search from backward_common

        all_result_arrays = set(re.findall(r'(?<=\n)(\w*?)(?=\s*?=)', forward_combined)) - all_tensors - redefined
        saved_arrays = []
        for _ in range(len(self.gradient)):
            saved_arrays.append(search_tensorarray(all_result_arrays, self.gradient[_]))
        saved_arrays.append(search_tensorarray(all_result_arrays, self.backward_common))

        saved_property = []
        saved_tensors_inclu_common = [saved_tensors[i] | saved_tensors[-1] for i in range(len(saved_tensors))]
        saved_arrays_inclu_common = [saved_arrays[i] | saved_arrays[-1] for i in range(len(saved_arrays))]

        for _ in range(len(self.gradient)):
            saved_property.append(
                {
                    i for i in re.findall(r'(?<!\w)(?:(?<!\.)\w+?)\.(?:[a-z]+?)(?![.(\w])', self.gradient[_])
                    if (i.split('.')[0] not in saved_tensors_inclu_common[_] | saved_arrays_inclu_common[_])
                       and (i.split('.')[0] in all_result_arrays | all_tensors)
                }
            )

        saved_property = sorted(saved_property[0].union(*saved_property[0:]))
        saved_tensors = sorted(saved_tensors[0].union(*saved_tensors[0:]))
        saved_arrays = sorted(saved_arrays[0].union(*saved_arrays[0:]))

        for p in saved_property:
            self.gradient = [re.sub(p, '_'.join(p.split('.')), i) for i in self.gradient]
        gradient_combined = [self.backward_common + '\n' * (len(self.backward_common) != 0) + i for i in self.gradient]

        return forward_combined, gradient_combined, forward_common, allow_inplace, inplace, outplace, inputs, outputs, \
               saved_tensors, saved_arrays, saved_property

    def inplace_write_to_text(self):
        c('inplace_precheck(xt0)')

        x0_in_grad = np.array([find_word('x0', i) is not None for i in self.gradient_combined])
        if x0_in_grad.all():
            x0_in_grad = 'all'
            condition = 'if ctx.requires_grad:'
        elif (~x0_in_grad).all():
            x0_in_grad = None
            condition = None
        else:
            x0_in_grad = np.argmax(x0_in_grad)
            condition = f'if ctx.needs_input_grad[{x0_in_grad}]:'
        self.x0_in_grad = x0_in_grad
        if condition:
            with indent(condition):
                c("ctx.params['copy'] = x0.copy()")
        if len(self.inplace) > 1:
            raise RuntimeError(f'{self.name}: inplace only support one output')
        inplace_code = self.inplace[0]

        if insert(inplace_code, 'y', 't') != inplace_code:
            raise RuntimeError(f'{self.name}: do not use variable y in inplace code')

        self.tensor_saved_inplace = None
        c(inplace_code)
        c('yt0 = inplace_update(xt0, ctx)')

    def outplace_write_to_text(self):
        for i, code in enumerate(self.outplace):
            c(code)
            if i > len(self.gradient) - 1:
                c(f"yt{i} = tt.tensor(y{i}, copy=False)")
            else:
                c(f"yt{i} = build_links(y{i}, grad_fn=ctx{'' if i == 0 else f', _output_idx={i}'})")

    def save_tensors_to_text(self, is_inplace):
        if is_inplace:
            if 'x0' not in self.saved_tensors:
                self.tensor_saved_inplace = False
                return
        saved_tensors = ['None' if (i == 'x0' and is_inplace) else i for i in self.saved_tensors]
        if saved_tensors:
            for i, tensor in enumerate(saved_tensors):
                if tensor == 'None':
                    continue
                tensor_in_grad = np.array([find_word(tensor, i) is not None for i in self.gradient_combined])
                if tensor_in_grad.all():
                    pass
                elif (~tensor_in_grad).all():
                    raise RuntimeError(f'{self.name}: bug: {tensor} not found in any gradient')
                else:
                    saved_tensors[i] = f'{tensor} if ctx.needs_input_grad[{np.argmax(tensor_in_grad)}] else None'
            saved_tensors = [insert(i, 'xy', 't') for i in saved_tensors]

            if self.allow_inplace and self.tensor_saved_inplace is False:
                Num_indents[0] -= 1
                c(f"ctx.save_for_backward({', '.join(saved_tensors)})")
                Num_indents[0] += 1
            else:
                c(f"ctx.save_for_backward({', '.join(saved_tensors)})")

    def load_tensors_to_text(self):
        if self.saved_tensors:
            c(f"{', '.join(self.saved_tensors)}{',' if len(self.saved_tensors) == 1 else ''} = ctx.saved_tensors")

    def save_arrays_to_text(self):
        saved_arrays = list(self.saved_arrays)
        if saved_arrays:
            for i, array in enumerate(saved_arrays):
                array_in_grad = np.array([find_word(array, i) is not None for i in self.gradient_combined])
                if array_in_grad.all():
                    pass
                elif (~array_in_grad).all():
                    raise RuntimeError(f'{self.name}: bug: {array} not found in any gradient')
                else:
                    saved_arrays[i] = f'{array} if ctx.needs_input_grad[{np.argmax(array_in_grad)}] else None'
            if len(saved_arrays) == 1:
                c(f"ctx.params['arrays'] = {saved_arrays[0]}")
            else:
                c(f"ctx.params['arrays'] = ({', '.join(saved_arrays)})")

    def load_arrays_to_text(self):
        if self.saved_arrays:
            if len(self.saved_arrays) == 1:
                c(f"{self.saved_arrays[0]} = ctx.params['arrays']")
            else:
                c(f"{', '.join(self.saved_arrays)} = ctx.params['arrays']")

    def save_property_to_text(self):
        if self.saved_property:
            if len(self.saved_property) == 1:
                c(f"ctx.params['property'] = {self.saved_property[0]}")
            else:
                c(f"ctx.params['property'] = ({', '.join(self.saved_property)})")

    def load_property_to_text(self):
        saved_property = ['_'.join(p.split('.')) for p in self.saved_property]
        if saved_property:
            if len(saved_property) == 1:
                c(f"{saved_property[0]} = ctx.params['property']")
            else:
                c(f"{', '.join(saved_property)} = ctx.params['property']")

    def load_grad_outputs(self):
        grads = ', '.join(f'g{i}' if find_word(f'g{i}', self.backward_combined) else '_' for i in
                          range(len(self.outputs)))
        c(f"{grads}{',' if len(self.outputs) == 1 else ''} = grad_outputs")

    def load_params_forward(self):
        for p in self.params:
            c(f"{p} = params['{p}']")

    def load_params_backward(self):
        if self.params:
            for p in self.params:
                if find_word(p, self.backward_combined):
                    c(f"{p} = ctx.params['{p}']")

    def load_xp(self, input_list):
        if find_word('xp', input_list):
            c("xp = ctx.xp")

    def need_x0_copy(self, list_of_string):
        if type(list_of_string) is str:
            list_of_string = [list_of_string]
        if self.inplace and 'x0' in self.saved_tensors:
            not_load = False
            for s in list_of_string:
                if not find_word('x0', s):
                    not_load = True
                    break
            if not not_load:
                with indent('if x0 is None:'):
                    c(f"x0 = ctx.params['copy']")
                    return True


def shorten_needs_input_grad():
    string_original = re.search(r'(##)([.\w\W]*?)(##)', to_save[0]).group(2)
    counter = len(re.findall(r'ctx.needs_input_grad\[', string_original))
    if counter >= 2:
        string = re.sub(r'\$\$req_grad\$\$\n', 'req_grad = ctx.needs_input_grad\n', string_original)
        string = re.sub(r'ctx.needs_input_grad\[', 'req_grad[', string)
    else:
        string = re.sub(r'\s*?\$\$req_grad\$\$', '', string_original)

    to_save[0] = re.sub(r'##([.\w\W]*?)##', string, to_save[0])


def generate_grad_func(input_fn, output_fn):
    # import os
    # if os.path.exists(f'{Path(__file__).parent}/{output_fn}'):
    #     return
    global to_save
    with open(rf'{Path(__file__).parent}/{input_fn}') as file:
        yml = yaml.load(file, Loader=yaml.FullLoader)
        c(yml['imports'])
        newline()
        c('"""')
        c('Auto-generated from grad_fcn_generator.py')
        c('Any changes to this file will NOT be kept during next import')
        c('Instead, make changes to grad_fcn_config.yaml to take effect')
        c('"""')
        newline(2)

        for name, config in yml.items():
            if name == 'imports':
                continue

            parse = Config(name, config)

            # forward
            with indent(f'class {name}(Function):'):
                parse.write_comment_to_text()
                with indent('##@staticmethod\ndef forward(ctx, *inputs, **params):'):
                    num_inputs = len(parse.inputs)
                    if num_inputs:
                        c(f"{', '.join(f'xt{_}' for _ in range(num_inputs))}{',' if num_inputs == 1 else ''} = inputs")
                        c(f"{', '.join(f'x{_}' for _ in range(num_inputs))} = {', '.join(f'xt{_}.data' for _ in range(num_inputs))}")
                    parse.load_params_forward()
                    c(f"$$req_grad$$")
                    parse.load_xp(parse.forward_combined)
                    c(parse.forward_common)
                    if parse.allow_inplace:
                        if parse.outplace:
                            with indent(f"if params['inplace']:"):
                                parse.inplace_write_to_text()
                                parse.save_tensors_to_text(is_inplace=True)
                            with indent('else:'):
                                parse.outplace_write_to_text()
                                parse.save_tensors_to_text(is_inplace=False)
                                # do not write anything on this line
                        else:
                            parse.inplace_write_to_text()
                            parse.save_tensors_to_text(is_inplace=True)
                    else:
                        parse.outplace_write_to_text()
                        parse.save_tensors_to_text(is_inplace=False)
                    parse.save_arrays_to_text()
                    parse.save_property_to_text()
                    outputs = sorted(insert(i, 'y', 't') for i in parse.outputs)
                    c(f"return {', '.join(outputs)}##")
                shorten_needs_input_grad()
                newline()

                with indent('@staticmethod\ndef backward(ctx, *grad_outputs):'):
                    parse.load_grad_outputs()
                    parse.load_tensors_to_text()
                    parse.load_arrays_to_text()
                    parse.load_property_to_text()
                    parse.load_params_backward()
                    x0_is_loaded = parse.need_x0_copy(parse.gradient_combined)
                    parse.load_xp(parse.backward_combined)
                    if len(parse.gradient) == 1:
                        c(parse.gradient[0])
                    else:
                        c(f"req_grad = ctx.needs_input_grad")
                        c(parse.backward_common)
                        c(f"{', '.join(f'grad{i}' for i in range(len(parse.gradient)))} = "
                          f"{', '.join('None' for _ in range(len(parse.gradient)))}")
                        for i in range(len(parse.gradient)):
                            order = parse.gradient_order[i]
                            with indent(f"if req_grad[{order}]:"):
                                code = parse.gradient[order]
                                if not x0_is_loaded:
                                    parse.need_x0_copy(code)
                                c(code)

                    c(f"return {', '.join(f'grad{i}' for i in range(len(parse.gradient)))}")
            newline(2)
        finished(output_fn)


if __name__ == '__main__':
    generate_grad_func('grad_fcn_config.yaml', 'grad_fcn.py')
