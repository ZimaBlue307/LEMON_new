import inspect
import os
import ast
import pickle

import astunparse
import copy
import sys
import astor
import json
import collections
import re

from utils import *
import mindspore
from mutation_utils_mindspore import LayerUtils



def find_module(module_dict, unique_name):
    '''
    :param module_dict:
    :param unique_name:
    :return: a list of module names, including all the modules having this node
    '''
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        return ["MindSporeModel"]
    else:
        module_list = ["MindSporeModel"]
        module_name = unique_names[0]
        init_func = module_dict["MindSporeModel"].body[0]
        for assign in init_func.body:
            if isinstance(assign, ast.Assign):
                target = assign.targets[0].attr
                if target == module_name:
                    module =  assign.value.func.id
                    module_list = deep_find_module(module_dict, module, unique_names[1:], module_list)
                    return module_list
        return None

def deep_find_module(module_dict, module_prefix, unique_names, module_list):
    # find Module based on module_prefix
    for key, module in enumerate(module_dict):
        if module_prefix == module:
            if len(unique_names) == 1:
                module_list.append(module)
                return module_list
            else:
                module_list.append(module)
                init_func = module_dict[module].body[0]
                for assign in init_func.body:
                    if isinstance(assign, ast.Assign):
                        target = assign.targets[0].attr
                        if target == unique_names[0]:
                            module = assign.value.func.id
                            module_list = deep_find_module(module_dict, module, unique_names[1:], module_list)
                            return module_list
    print("ERROR: Not find corresbonding module {}".format(unique_names))
    return None

def get_name(data_item):
    unique_name, module_name = data_item[0], data_item[1]
    module_names = module_name.split(".")
    module_name = module_names[-1]
    if len(module_names) > 1:
        unique_name = '.'.join(module_names[:-1]) + '.' + unique_name
    return unique_name, module_name


def get_copy_name(module_list):
    if not module_list:
        return None
    copy_name = list()
    for i in range(len(module_list)):
        copy_name.append(0)
    return copy_name


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
        copy_num = get_copy_name(module_name)
        unique_name, operator_name = get_name(data_item)
        node.set_uniquename(unique_name)
        node.set_operator_name(operator_name)
        node.set_module(module_name)
        node.set_copy_num(copy_num)
        node.set_input(data_item[3])
        return_list = get_model_index(data_item[3], analyzed_data)
        node.set_input(return_list)

        table.add_node(node)
    output_list = get_model_output_index(analyzed_data)
    for index in table.nodeList.keys():
        ast_index = get_ast_index(table, index)
        table.nodeList[index].set_ast_index(ast_index)
        table.nodeList[index].set_output(output_list[index])
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

def get_model_output_index(analyzed_data):
    for i, data_element in enumerate(analyzed_data):
        return_list = []
        output_name = data_element[0]
        op_tuple = tuple(data_element[1].split("."))
        op_prefix = '.'.join(op_tuple[:-1])
        if 'ast' in data_element[1]: #处理一些特殊情况；
            input_search = output_name
        elif len(op_prefix) != 0:
            input_search = op_prefix + "." + output_name
        else:
            input_search = output_name
        #默认analyzed_data的最后一条元素是最终的输出；
        if data_element == analyzed_data[-1]:
            return_list.append(-2) #表示最终输出张量；
        #先考虑相同class之内的；
        for i, element in enumerate(analyzed_data):
            input_list = element[-1]
            for j, input in enumerate(input_list):
                if input_search == input:
                    return_list.append(i)
                    break
                else:
                    continue
            if len(return_list) != 0:
                break
        data_element.append(return_list)

    #再考虑class跳转出去的，以及其他特殊情况.初步打算倒着遍历；
    length = len(analyzed_data)
    for i in range(length):
        data_element = analyzed_data[length - i - 1]
        return_list = []
        if len(data_element[-1]) == 0:
            op_tuple = tuple(data_element[1].split("."))
            op_prefix = '.'.join(op_tuple[:-1])
            for j in range(length):
                element = analyzed_data[length - j - 1]
                entire_op_name = element[1]
                if (entire_op_name == op_prefix) and (element[-1] != []):
                    return_list = element[-1]
                    break
            data_element[-1] = return_list
    # 最后可能还要考虑其他的特殊情况，需要不断补充
    return_list = dict()
    for i, dataitem in enumerate(analyzed_data):
        tmp = dataitem[4]
        return_list[i] = tmp
    return return_list

def set_copy_module_name(module_name, index):
    module_name = module_name.split("_")[0]
    return module_name + '_' + str(index)

def same_module_list(table, index, module_list):
    indices = list()
    # two judge:
    # 1. node module should be the same
    # 2. unique name prefix should be the same

    prefix = table.nodeList[index].get_prefix()
    for i in range(table.node_list_len()):
        # compare every node with module_list, if same, save it in indices
        # if collections.Counter(table.nodeList[i].node_module) == collections.Counter(module_list):
        #     indices.append(i)
        if prefix == table.nodeList[i].get_prefix():
            indices.append((i))

    return indices

def search_init_ms_class(table, index): # newly added by hyr
    node = table.nodeList[index]
    module_list = node.node_module
    prefix = node.unique_name.split(".")[:-1]
    prefix.append(node.operator_name)
    return_list = list()
    for i in range(len(module_list)):
        for item in table.ast.body:
            if isinstance(item, ast.ClassDef) and item.name == module_list[i]:
                init_func = item.body[0]
                for j in range(len(init_func.body)):
                    if isinstance(init_func.body[j], ast.Assign) and init_func.body[j].targets[0].attr == prefix[i]:
                        func_node = init_func.body[j].value.func
                        if isinstance(func_node, ast.Attribute):
                            attr_str = func_node.attr
                            if hasattr(func_node.value, 'id'):
                                id_str = func_node.value.id
                                join_list = [id_str, attr_str]
                                ms_class_str = ".".join(join_list)
                            else:
                                id_str = func_node.value.value.id
                                value_attr = func_node.value.attr
                                join_list = [id_str, value_attr, attr_str]
                                ms_class_str = ".".join(join_list)
                        elif isinstance(func_node, ast.Name):
                            ms_class_str = func_node.id
                        else:
                            print("this assign node belongs to an other class: {}".format(type(func_node)))
                            continue
                        return_list.append(ms_class_str)
    return return_list

def search_init_statement(table, index): #search_init_statement
    node = table.nodeList[index]
    module_list = node.node_module
    prefix = node.unique_name.split(".")[:-1]
    prefix.append(node.operator_name)
    return_list = list()
    for i in range(len(module_list)):
        for item in table.ast.body:
            if isinstance(item, ast.ClassDef) and item.name == module_list[i]:
                init_func = item.body[0]
                for j in range(len(init_func.body)):
                    if isinstance(init_func.body[j], ast.Assign) and init_func.body[j].targets[0].attr == prefix[i]:
                        return_list.append(j)
    return return_list

def return_construct_op_name(node):
    if isinstance(node, ast.BinOp):
        node_type = type(node.op)
        return BinOpTable[str(node_type)]
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Name):
        return node.id
    

def search_construct_statement(table, index):
    node = table.nodeList[index]
    module_list = node.node_module
    prefix = node.unique_name.split(".")[:-1]
    prefix.append(node.operator_name)
    operator_output_name = node.unique_name.split(".")[-1]

    return_list = list()

    for i in range(len(module_list)):
        for item in table.ast.body:
            if isinstance(item, ast.ClassDef) and item.name == module_list[i]:
                construct_func = item.body[1]
                for j in range(len(construct_func.body)):
                    sub_node = construct_func.body[j]
                    if isinstance(sub_node, ast.Assign):
                        if isinstance(sub_node.value, ast.Call):
                            return_name = return_construct_op_name(sub_node.value.func)
                        elif isinstance(sub_node.value, ast.BinOp):
                            return_name = return_construct_op_name(construct_func.body[j].value)
                        if i == len(module_list) - 1:
                            if return_name == prefix[i] and sub_node.targets[0].id == operator_output_name:
                                # print("find!")
                                return_list.append(j)
                                break
                        else:
                            if return_name == prefix[i]:
                                # print("find module!")
                                return_list.append(j)
                                break
    return return_list

def get_ast_index(table, index):
    init_list = search_init_statement(table, index)
    cons_list = search_construct_statement(table, index)
    if not len(init_list) == len(cons_list):
        print("error happen! {}".format(index))
        return None
    return_list = list()
    for i in range(len(init_list)):
        tmp = [init_list[i], cons_list[i]]
        return_list.append(tmp)
    return return_list

def copy_module(table, index, param_dict):
    '''
    copy the module related the index, insert them in the model_ast
    :param table:

    :return:
    '''
    # get the module list
    target_node = table.nodeList[index]
    prefix = target_node.unique_name.split(".")[:-1]
    prefix = ".".join(prefix)
    module_list = target_node.node_module
    print(module_list)
    new_module_list = ['MindSporeModel']
    add_list = list()
    for i in range(1, len(module_list)):
        module_name = module_list[i]

        for item in table.ast.body:
            if isinstance(item, ast.ClassDef) and item.name == module_name:
                tmp_ast = copy.deepcopy(item)
                # modify table info

                table.nodeList[index].copy_num[i] += 1
                new_module_name = set_copy_module_name(tmp_ast.name, table.nodeList[index].copy_num[i])
                new_module_list.append(new_module_name)
                # table.nodeList[index].node_module[i] = new_module_name

                # add copy module ast
                tmp_ast.name = new_module_name
                super_node = tmp_ast.body[0].body[0]
                if isinstance(super_node, ast.Expr):
                    super_param = super_node.value.func.value.args
                    super_param[0].id = new_module_name
                add_list.append(tmp_ast)

    # modify table info, including other indexes that have the same module list
    indices = same_module_list(table, index, module_list)
    # every index need change node_module and copy_num
    # new_module_list = table.nodeList[index].node_module
    new_copy_num = table.nodeList[index].copy_num
    for i in indices:
        table.nodeList[i].node_module = new_module_list
        table.nodeList[i].copy_num = new_copy_num

    print(len(add_list))
    # add copied ast
    for module_ast in add_list:
        # add every ast at the end of model ast.body
        length = len(table.ast.body)
        table.ast.body.insert(length, module_ast)

    # update ast info
    # modify every states in indices
    for i in range(len(new_module_list) - 1):
        for j in range(len(table.ast.body)):
            if isinstance(table.ast.body[j], ast.ClassDef) and table.ast.body[j].name == module_list[i]:
                init_node = table.ast.body[j].body[0].body[target_node.ast_index[i][0]].value
                # init_node.value.func = new_module_list[i+1]
                statement = astunparse.unparse(init_node)
                node_split = statement.split("(")
                node_split[0] = new_module_list[i+1]
                new_statement = "(".join(node_split)
                new_node = ast.parse(new_statement).body[0].value
                table.ast.body[j].body[0].body[target_node.ast_index[i][0]].value = new_node
                print(table.ast.body[j].body[0].body[target_node.ast_index[i][0]].value)

    # for i in indices:
    #     # change init func info
    #

    # copy param
    # new_prefix = target_node.unique_name.split(".")[:-1]
    # new_prefix = ".".join(new_prefix)
    # added_param = dict()
    # for key in enumerate(param_dict):
    #     if prefix in key:
    #         param = copy.deepcopy(param_dict[key])
    #         new_op_name = key.replace(prefix, new_prefix)
    #         added_param[new_op_name] = param
    # param_dict = param_dict + added_param

    return table, param_dict

def insert_node(table, index, new_node_name, **kwargs):
    '''
    :param table:
    :param index:
    :param new_node_name:
    :param kwargs:
    :return:
    '''
    target_node = table.nodeList[index]
    # if the module num is more than mindsporeModel, we need copy the module
    if len(target_node.node_module) > 1:
        table, model_ast = copy_module(table=table, index=index, param_dict=param_dict)

    # insert the node after the index
    layer_utils = LayerUtils()
    if new_node_name in layer_utils.available_model_level_layers.keys():
        insert_str = layer_utils.available_model_level_layers[new_node_name](**kwargs)
        # get op_name
        op_name = "addNode_" + table.add_node_num
        out_name = "opt_addNode_" + table.add_node_num
        table.add_node_num += 1
        # get full inser str
        insert_str = op_name + " = " + insert_str
        insert_node = ast.parse(insert_str).body[0]

        #insert the node
        # in ast, direct add node after the statement
        for module in table.ast.body:
            if isinstance(module, ast.ClassDef) and module.name == target_node.node_module[-1]:
                assert len(target_node.input_list) == 1
                ast_index = target_node.ast_index[-1]
                init_func = module.body[0]
                init_func.body.insert(ast_index[0], insert_node)



        node_len = table.node_list_len()
        insert_index = node_len
        #默认插入的module位置和上一行相同




    else:
        print("{} not implemented!".format(new_node_name))

def test_kwargs(table, index, new_node_name, **kargs):
    print("index: ", index)

#
# def replace_node(table, index, new_node):
#     raise NotImplementedError

def delete_node(table, index, param_dict):
    '''
    :param table:
    :param index:
    :param model_ast:
    :param param_dict:
    :return:
    '''
    target_node = table.nodeList[index]
    # if the module num is more than mindsporeModel, we need copy the module
    if len(target_node.node_module) > 1:
        table, param_dict = copy_module(table=table, index=index, param_dict=param_dict)
    # delete the index
    # deal with the ast
    # get the input nodes and output nodes of index node
    inputs = target_node.input_list
    outputs = target_node.output_list

    # only delete the final op in the last module
    for module in table.ast.body:
        if isinstance(module, ast.ClassDef) and module.name == target_node.node_module[-1]:
            #we suppose that the delete node have only one input
            assert(len(target_node.input_list) == 1)
            ast_index = target_node.ast_index[-1]
            del module.body[0].body[ast_index[0]]
            if ast_index[1] == len(module.body[1].body) - 2:
                # change the final return name
                input_node_index = table.nodeList[inputs[0]].ast_index[-1][1]
                return_name = module.body[1].body[input_node_index].targets[0].id
                module.body[1].body[-1].value.id = return_name
            elif ast_index[1] == 0:
                # we suppose all delete node has only one input
                input_param = module.body[1].args.args[1].arg
                delete_output_name = module.body[1].body[ast_index[1]].targets[0].id
                for output in outputs:
                    output_index = table.nodeList[output].ast_index[-1][1]
                    output_node = module.body[1].body[output_index]
                    out_str = astunparse.unparse(output_node)
                    if delete_output_name in out_str:
                        pattern = re.compile(delete_output_name)
                        out_str = pattern.sub(input_param, out_str)
                    else:
                        print("delete first statement error happen!")
                    new_node = ast.parse(out_str).body[0]
                    module.body[1].body[output_index] = new_node
            del module.body[1].body[target_node.ast_index[-1][1]]
    # every node in inputs del the index in outputs
    for i in inputs:
        node = table.nodeList[i]
        node.output_list.remove(index)
        node.output_list = node.output_list + outputs
    del table.nodeList[index]
    # for i in outputs:
    #     node = table.nodeList[i]
    #     node.input_list.remove(index)
    for index in table.nodeList.keys():
        ast_index = get_ast_index(table, index)
        table.nodeList[index].set_ast_index(ast_index)

    return table, param_dict


def pickle_save(model_path):
    model_ast = astor.parse_file(model_path)

    # analyzed_data =
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item

    # construct our table
    table = construct_table(model_ast, analyzed_data, module_dict)
    table_save_tuple = tuple(model_path.split("/"))
    table_name = table_save_tuple[-1]
    table_name = tuple(table_name.split("."))[0]
    table_save_path = '/'.join(table_save_tuple[:-1]) + "/" + table_name + '_table.pkl'
    with open(table_save_path, "wb") as file1:
        pickle.dump(table, file1)
    with open(table_save_path, "rb") as file2:
        new_table = pickle.load(file2)
    return new_table

if __name__ == '__main__':
    # get python file and ckpt file
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    ckpt_path = f'../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt'
    model_ast = astor.parse_file(model_path)
    param_dict = mindspore.load_checkpoint(ckpt_path)
    # print(param_dict.keys())
    # result_dict = get_code(model_ast)

    #construct a dict including module index in model_ast.body
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item
    # print(module_dict)

    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)

    print(len(analyzed_data))
    table = construct_table(model_ast, analyzed_data, module_dict)
    
    
    # print(len(table.nodeList))
    # table.print()
    # index = 1
    # # returnlist = get_ast_index(table, index)
    # # table, param_dict = delete_node(table=table, index=index, param_dict=param_dict)
    # new_node_name = "conv_2d"
    # kwargs = {"in_channels":2, "out_channels": 3, "kernel_size":(2, 2)}
    # table, param_dict = insert_node(table, index, new_node_name, **kwargs)
    # # test_kwargs(table=table, index=index, )
    # table.print()
    # table.save_ast(save_path="result.py")
