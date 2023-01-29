import json
import ast
import astunparse
import sys
import astor
import os
import pickle
import numpy as np
from mindspore import Tensor
import mindspore
import scripts.tools.utils
from scripts.logger.lemon_logger import Logger
from scripts.mutation.model_mutation_operators import *

main_logger = Logger()

class PrintVisitor(ast.NodeVisitor):

    def __init__(self):
        super(PrintVisitor, self).__init__()

    def visit_Assign(self, node):
        '''
        for every assign statement, add a print node after it
        print the following thing
        1. if it is not a module, print the name (e.g. conv1d) and the shape
        2. if it is a module, print the module name
        :param node:
        :return:
        '''

        self.generic_visit(node)
        return node


    def visit_FunctionDef(self, node):
        if 'construct' in node.name:
            # add a print node after every assign node
            global_str = 'global save'
            global_node = ast.parse(global_str).body[0]
            node.body.insert(0, global_node)


            i = 0
            flag = True
            while flag:
                sub_node = node.body[i]
                if isinstance(sub_node, ast.Assign):
                    if isinstance(sub_node.value, ast.Call): # check the assign node is using a operator, not just a parameter
                        target = sub_node.targets[0].id
                        print_str = None
                        if 'module' in target:
                            # if it is a module, change the statement with a construct statement
                            # print_str = "print('{} is a module')\n".format(target)
                            node_str = astunparse.unparse(sub_node)
                            # print(node_str)
                            node_strs = node_str.split('(')
                            new_str = ''
                            for j in range(len(node_strs)):
                                if j == 0:
                                    new_str = node_strs[j] + '.construct'
                                else:
                                    new_str += '('
                                    new_str += node_strs[j]
                            print(new_str)
                            new_node = ast.parse(new_str).body[0]
                            node.body[i] = new_node
                            module_name = sub_node.value.func.attr
                            args = sub_node.value.args
                            arg_list = list()
                            for arg in args:
                                if isinstance(arg, ast.Name):
                                    arg_list.append(arg.id)
                            pre_stat = "save.append(['{}', '{} start', {}])".format(module_name, module_name, arg_list)
                            aft_stat = "save.append(['{}', '{} end', {}, '{}'])".format(module_name, module_name, arg_list, target)
                            print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target, sub_node.value.func.attr, target, arg_list)
                            pre_node = ast.parse(pre_stat).body[0]
                            aft_node = ast.parse(aft_stat).body[0]
                            print_node = ast.parse(print_str).body[0]
                            node.body.insert(i, pre_node)
                            node.body.insert(i+2, aft_node)
                            node.body.insert(i+2, print_node)
                            # print(astunparse.unparse(node))
                            i = i+2
                        else:
                            # get shape
                            if isinstance(sub_node.value, ast.Call):
                                if isinstance(sub_node.value.func, ast.Attribute):
                                    args = sub_node.value.args
                                    arg_list = list()
                                    for arg in args:
                                        if isinstance(arg, ast.Name):
                                            arg_list.append(arg.id)
                                        elif isinstance(arg, ast.Attribute):
                                            arg_list.append(arg.attr)
                                    print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target, sub_node.value.func.attr, target, arg_list)
                                elif isinstance(sub_node.value.func, ast.Call):
                                    args = sub_node.value.args
                                    arg_list = list()
                                    for arg in args:
                                        if isinstance(arg, ast.Name):
                                            arg_list.append(arg.id)
                                        elif isinstance(arg, ast.Attribute):
                                            arg_list.append(arg.attr)
                                    print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target,
                                                                                             sub_node.value.func.func.attr,
                                                                                             target, arg_list)
                                # sub_node_str = astunparse.unparse(sub_node)
                                # print_str = sub_node_str  + print_str + '\n'
                                print_ast = ast.parse(print_str)
                                print_node = print_ast.body[0]
                                print_ast_str = ast.dump(print_node)
                                node.body.insert(i + 1, print_node)
                                # i += 1
                                # print(print_ast_str)
                            else:
                                print("this line is not a attribute node!")

                    elif isinstance(sub_node.value, ast.BinOp):
                        print("++++++++++Get in binop branch!++++++++++")
                        # if the node is binOp, we need add both left name and right name
                        # into the input list
                        left = sub_node.value.left
                        right = sub_node.value.right
                        target = sub_node.targets[0].id

                        def get_ast_name(node):
                            if isinstance(node, ast.Name):
                                return node.id
                            elif isinstance(node, ast.Attribute):
                                return node.attr

                        left_name, right_name = get_ast_name(left), get_ast_name(right)
                        arg_list = [left_name, right_name]
                        print_str = "save.append(['{}', \"{}\", {}.shape, {}])".format(target,
                                                                                     str(type(sub_node.value.op)),
                                                                                     target, arg_list)
                        print_ast = ast.parse(print_str)
                        print_node = print_ast.body[0]
                        print_ast_str = ast.dump(print_node)
                        node.body.insert(i + 1, print_node)


                elif isinstance(sub_node, ast.Return):
                    flag = False
                #     print_str = "print('end')\n"
                #     print_ast = ast.parse(print_str)
                #     print_node = print_ast.body[0]
                #     print_ast_str = ast.dump(print_node)
                #     node.body.insert(i, print_node)


                i = i+1
        ast.fix_missing_locations(node)
        self.generic_visit(node)
        return node



BinOpTable = {
    "<class '_ast.Add'>": "add",
    "class '_ast.Sub'": "sub",
    "class '_ast.Mult'": "mult",
    "class '_ast.Div'": "div",
    "class '_ast.FloorDiv'": "floordiv",
    "class '_ast.Mod'": "mod",
    "class '_ast.Pow'": "pow",
    "class '_ast.LShift'": "lshift",
    "class '_ast.RShift'": "rshift",
    "class '_ast.BitOr'": "bitor",
    "class '_ast.BitXor'": "bitxor",
    "class '_ast.BitAnd'": "bitand",
    "class '_ast.MatMult'": "matmult"
}

class ShapeUitls(object):

    def __init__(self):
        super(ShapeUitls, self).__init__()

    @staticmethod
    def modify_ast(source_ast, output_path):
        '''
        Given the ast of source code, and theoutput path, modify source code and save it
        :param source_ast:
        :param output_path:
        :return:
        '''

        # add a list to save the shape
        i = 0
        for i in range(len(source_ast.body)):
            if not isinstance(source_ast.body[i], ast.Import) and not isinstance(source_ast.body[i], ast.ImportFrom):
                break
        list_str = 'save = list()'
        list_node = ast.parse(list_str).body[0]
        source_ast.body.insert(i, list_node)

        # add import json
        json_node = ast.parse('import json').body[0]
        source_ast.body.insert(i, json_node)

        visitor = PrintVisitor()
        visitor.visit(source_ast)

        # add print in the last construct function
        last_construct = source_ast.body[-1].body[-1]
        # print(astunparse.unparse(last_construct))
        print_str1 = "with open('{}', 'w') as f:\n   json.dump(save, f)".format(output_path)
        print_node = ast.parse(print_str1)
        print_node1 = print_node.body[0]
        # print_node2 = print_node.body[1]

        last_construct.body.insert(-1, print_node1)
        # last_construct.body.insert(-1, print_node2)

        # print_str = "print(save)"
        # print_node = ast.parse(print_str).body[0]
        # last_construct.body.insert(-1, print_node)
        return source_ast

    @staticmethod
    def module_analyze(shape_list, target_list, module_name, index, prefix_list=None):
        while index < len(shape_list):
            item = shape_list[index]
            if 'module' in item[0]:
                if 'start' in item[1]:
                    tmp_name = module_name + '.' + item[0]
                    tmp_prefix = list()
                    if not prefix_list:
                        for ele in item[2]:
                            tmp_prefix.append(module_name + '.' + ele)
                    else:
                        tmp_prefix = prefix_list
                    target_list, index = ShapeUitls.module_analyze(shape_list, target_list, tmp_name, index + 1, tmp_prefix)
                elif 'end' in item[1]:
                    print('module finished! index come to {}'.format(index))
                    # module_name, end_info, arg_list, module_output_name
                    return target_list, index
                else:
                    # 回退一个module
                    modules = module_name.split(".")
                    return_module_name = ''
                    if len(modules) == 1:
                        node_name = module_name
                    else:
                        return_module_name = '.'.join(modules[:-1])
                        node_name = return_module_name + '.' + item[1]
                    new_list = list()
                    if not prefix_list:
                        for ele in item[3]:
                            if return_module_name == '':
                                new_list.append(ele)
                            else:
                                new_list.append(return_module_name + '.' + ele)
                    else:
                        for ele in prefix_list:
                            new_list.append(ele)
                        prefix_list = None
                    target_list.append([item[0], node_name, item[2], new_list])
            else:
                # print(item[1])
                if 'class' in item[1]:
                    op_name = module_name + '.' + BinOpTable[item[1]]
                else:
                    op_name = module_name + '.' + item[1]
                new_list = list()
                if not prefix_list:
                    for ele in item[3]:
                        new_list.append(module_name + '.' + ele)
                else:
                    for ele in prefix_list:
                        new_list.append(ele)
                    prefix_list = None
                target_list.append([item[0], op_name, item[2], new_list])
            index += 1
        print('module finished! index come to {}'.format(index))
        return target_list, index

    @staticmethod
    def model_analyze(model_dir, backend_name, cfg_name):
        # model_dir example: "origin_model/ms_model/resnet20_cifar100"
        model_prefix = model_dir.split("/")[-1]
        model_path = model_prefix + '.py'
        tmp_output = model_prefix + '_modified.py'
        model_path = os.path.join(model_dir, model_path)
        param_path = model_prefix + '.ckpt'
        param_path = os.path.join(model_dir, param_path)

        network_ast = astor.code_to_ast.parse_file(model_path)
        output_path = os.path.join(model_dir, 'inter_result.json')
        modified_ast = ShapeUitls.modify_ast(network_ast, output_path)
        modified_model_path = os.path.join(model_dir, tmp_output)
        with open(modified_model_path, 'w') as f:
            f.write(astunparse.unparse(modified_ast))

        def get_opt_cls_name():
            mindsporemodel_var = "MindSporeModel"
            return mindsporemodel_var

        _opt_cls_name = get_opt_cls_name()
        sys.path.append(model_dir)
        modified_model_object = __import__(tmp_output[:-3])
        modified_model_object = getattr(modified_model_object, _opt_cls_name)
        modified_model_object = modified_model_object()
        param_dict = mindspore.load_checkpoint(param_path)
        mindspore.load_param_into_net(modified_model_object, param_dict)

        # 获得数据集的一个切片
        dataset, dataset_name = utils.DataUtils.get_data_by_exp_with_bk(model_prefix, 4, backend_name, cfg_name)
        dataset = dataset.batch(batch_size=4)
        for i, d in enumerate(dataset.create_dict_iterator()):
            if i==0:
                test_data = d["image"]
                main_logger.info("INFO: test_data shape {}".format(tuple(test_data.shape)))
                input = Tensor(np.random.uniform(0, 1, tuple(test_data.shape)), dtype=mindspore.float32)
                break
        output = modified_model_object.construct(input)

        with open(output_path, 'r') as f:
            shape_list = json.load(f)
        # print(shape_list)
        analyzed_shape = list()
        i = 0
        while i < len(shape_list):
            item = shape_list[i]
            if 'module' in item[0]:
                print('start analyze module!')
                module_name = item[0]
                i += 1
                # iteraly analyze the module
                print(len(item))
                analyzed_shape, i = ShapeUitls.module_analyze(shape_list, analyzed_shape, module_name, i, item[2])
            else:
                if 'class' in item[1]:
                    item[1] = BinOpTable[item[1]]
                analyzed_shape.append(item)
            i += 1
        main_logger.info("INFO: save inter result in {}".format(os.path.join(model_dir, "analyzed_data.json")))
        with open(os.path.join(model_dir, "analyzed_data.json"), 'w') as f:
            json.dump(analyzed_shape, f)

        return os.path.join(model_dir, "analyzed_data.json")

