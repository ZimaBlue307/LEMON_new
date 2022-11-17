import inspect
import os
import ast
import astunparse
import copy
import sys
import astor
import json


import mindspore

class Node(object):
    def __init__(self, index, unique_name, shape, operator_name = None, node_module=None, input_list=None, output_list=None, output_name=None):
        self.index = index
        self.unique_name = unique_name # module name + output name
        self.operator_name = operator_name
        self.node_module = node_module
        self.input_list = input_list
        self.output_list = output_list
        self.shape = shape
        self.output_name = output_name

    def set_uniquename(self, unique_name):
        self.unique_name = unique_name

    def set_operator_name(self, operator_name):
        self.operator_name = operator_name

    def set_input(self, input_list):
        self.input_list = input_list

    def set_output(self, output_list):
        self.output_list = output_list

    def set_module(self, module):
        self.node_module = module


class Table(object):
    def __init__(self, model_ast):
        self.nodeList = dict()
        self.ast = model_ast

    def add_node(self, node):
        self.nodeList[node.index] = node

    def print_nodelist(self):
        for item in self.nodeList:
            item = self.nodeList[item]
            print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module, '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list)

def find_module(module_dict, unique_name):
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        return "MindSporeModel"
    else:
        module_name = unique_names[0]
        init_func = module_dict["MindSporeModel"].body[0]
        for assign in init_func.body:
            if isinstance(assign, ast.Assign):
                target = assign.targets[0].attr
                if target == module_name:
                    module =  assign.value.func.id
                    return deep_find_module(module_dict, module, unique_names[1:])
        return None

def deep_find_module(module_dict, module_prefix, unique_names):
    # find Module based on module_prefix
    for key, module in enumerate(module_dict):
        if module_prefix == module:
            if len(unique_names) == 1:
                return module
            else:
                init_func = module_dict[module].body[0]
                for assign in init_func.body:
                    if isinstance(assign, ast.Assign):
                        target = assign.targets[0].attr
                        if target == unique_names[0]:
                            module = assign.value.func.id
                            return deep_find_module(module_dict, module, unique_names[1:])
    print("ERROR: Not find corresbonding module {}".format(unique_names))
    return None

def get_name(data_item):
    unique_name, module_name = data_item[0], data_item[1]
    module_names = module_name.split(".")
    module_name = module_names[-1]
    if len(module_names) > 1:
        unique_name = '.'.join(module_names[:-1]) + '.' + unique_name
    return unique_name, module_name

def construct_table(model_ast, analyzed_data, module_dict):


    class MyNodeVisitor(ast.NodeVisitor):
        def __init__(self):
            super(MyNodeVisitor, self).__init__()

        def visit_Assign(self, node: ast.Assign):
            pass

    table = Table(model_ast)
    for i, data_item in enumerate(analyzed_data):
        node = Node(index=i, unique_name=data_item[1], shape=data_item[2], output_name=data_item[0])

        module_name = find_module(module_dict, node.unique_name)
        unique_name, operator_name = get_name(data_item)
        node.set_uniquename(unique_name)
        node.set_operator_name(operator_name)
        node.set_module(module_name)
        node.set_input(data_item[3])
        return_list = get_model_index(data_item[3], analyzed_data)
        node.set_input(return_list)
        table.add_node(node)

    return table

def get_model_index(input_list, analyzed_data):
    """
    for example,
    input_list: ['opt_conv2d_51', 'module3_1_opt'] or ['module5_0.module0_0.opt_batchnorm2d_0']
    Each input_list can be obtained from the input element in analyzed_data[i]
    analyzed_data is the same as file analyzed_data.json;
    return a return_list, return_list[i] is the index of input_list[i]; and len(return_list) equals to len(input_list)
    """
    return_list = list()
    for i, input in enumerate(input_list):
        if 'input' in input:
            # print("This is the input.")
            return_list.append(-1)
        else:
            input_tuple = tuple(input.split("."))
            input_name = input_tuple[-1] #最后一位是输入的name
            input_prefix = input.rstrip("." + input_name)  #前缀用来筛选，防止出现相同的name
            # print(input_name)
            # print(input_prefix)
            # print("===========")
            for i, element in enumerate(analyzed_data):
                if (input_name != 'x') and (input_name == element[0]) and (input_prefix in element[1]):
                    return_list.append(i)
                    break
                elif (input_name == 'x') and (input_prefix in element[1]): #往上找到第一个的input的index;
                    return_list.append(i)
                    break
                else:
                    continue
    return return_list

def insert_node():
    raise NotImplementedError

def replace_node(table, index, new_node):
    raise NotImplementedError

def delete_node(table, index):
    raise NotImplementedError

if __name__ == '__main__':
    # get python file
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    model_ast = astor.parse_file(model_path)
    # result_dict = get_code(model_ast)

    #construct a dict including module index in model_ast.body
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item
    print(module_dict)

    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)

    print(len(analyzed_data))
    table = construct_table(model_ast, analyzed_data, module_dict)
    print(len(table.nodeList))
    table.print_nodelist()