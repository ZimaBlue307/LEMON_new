import inspect
import os
import ast
import pickle
import math
from typing import *

import astunparse
import copy
import sys
import astor
import json
import collections
import re
import numpy as np
from utils import *
import mindspore
# from scripts.logger.lemon_logger import Logger 这会报错诶
from mutation_utils_mindspore import LayerUtils
from layer_matching_mindspore import LayerMatching, ResizeBilinear

# mylogger = Logger()

def set_copy_module_name(module_name, index):
    module_name = module_name.split("_")[0]
    return module_name + '_' + str(index)


def same_module_list(table, index):
    indices = list()
    # two judge:
    # 1. node module should be the same
    # 2. unique name prefix should be the same
    prefix = table.nodeList[index].get_prefix()
    for i in range(table.node_list_len()):
        if prefix == table.nodeList[i].get_prefix():
            indices.append((i))
    return indices


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
    indices = same_module_list(table, index)
    # every index need change node_module and copy_num
    # new_module_list = table.nodeList[index].node_module
    new_copy_num = table.nodeList[index].copy_num
    for i in indices:
        table.nodeList[i].node_module = new_module_list
        table.nodeList[i].copy_num = new_copy_num
    # print(len(add_list))
    # add copied ast
    for module_ast in add_list:
        # add every ast at the end of model ast.body
        length = len(table.ast.body)
        table.ast.body.insert(length, module_ast)
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
    return table, param_dict


def _assert_indices(mutated_layer_indices: List[int] , depth_layer: int):#done
    assert max(mutated_layer_indices) < depth_layer,"Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0,"Min index should be greater than or equal to zero"


def _MLA_model_scan(irtable, new_layers, mutated_layer_indices=None):
    layer_matching = LayerMatching()# need to change file LayerMatching
    nodeList = irtable.nodeList
    positions_to_add = np.arange(len(nodeList) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(nodeList))
    insertion_points = {}
    available_new_layers = [layer for layer in layer_matching.layer_concats.keys()] if new_layers is None else new_layers
    for node_index in nodeList.keys():
        node = nodeList[node_index]
        operator_name = node.operator_name
        output_shape = node.shape
        # print(operator_name, ": ", output_shape, "; length: ", len(output_shape))
        if 'softmax' in operator_name.lower():
            break
        if node_index in positions_to_add:
            for available_new_layer in available_new_layers:
                if layer_matching.input_legal[available_new_layer](output_shape):
                    if node_index not in insertion_points.keys():
                        insertion_points[node_index] = [available_new_layer]
                    else:
                        insertion_points[node_index].append(available_new_layer)
    return insertion_points

def insert_node(table, param_dict, index, new_node_name, **kwargs):
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


def insert_multi_node(table, param_dict, new_layers = None, mutated_layer_indices=None, **kwargs):
    '''
    :param table:
    :param index:
    :param new_node_name:
    :param kwargs:
    :return:
    '''
    layer_matching = LayerMatching()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    insertion_points = _MLA_model_scan(table, new_layers, mutated_layer_indices)
    #insertion_points的key是index，value是相应的layer的string;
    if len(insertion_points.keys()) == 0:
        print('no appropriate layer to insert')
        return None
    #这里暂时缺少“哪里可以插入某层”的logger信息。
    nodeList = table.nodeList
    layers_index_avaliable = list(insertion_points.keys()) #所有可以进行MLA的层
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))] #从所有可以MLA的层中挑一个
    all_new_layers_str = insertion_points[layer_index_to_insert] #获得一层中所有可以插入的layer
    layer_name_to_insert = all_new_layers_str[np.random.randint(0, len(all_new_layers_str))] #挑一个决定插入的layer
    print("choose to insert {} after {}.".format(layer_name_to_insert, nodeList[layer_index_to_insert].operator_name))
    
    #insert new layers
    insert_node(table, param_dict, layer_index_to_insert, layer_name_to_insert, **kwargs)
    
    
def test_insert_node():
    ckpt_path = f'../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt'
    param_dict = mindspore.load_checkpoint(ckpt_path)
    new_node_name = "conv_2d"
    index = 1
    kwargs = {"in_channels":2, "out_channels": 3, "kernel_size":(2, 2)}
    table, param_dict = insert_node(table, param_dict, index, new_node_name, **kwargs)
    table.print()
    table.save_ast(save_path="result.py")

def test_insert_multi_node():
    #get model
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    ckpt_path = f'../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt'
    param_dict = mindspore.load_checkpoint(ckpt_path)
    # from origin_model.ms_model.resnet20_cifar100.resnet20_cifar100_origin import MindSporeModel
    # resnet20 = MindSporeModel()
    # mindspore.load_param_into_net(resnet20, param_dict)
    
    # get table
    model_ast = astor.parse_file(model_path)
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item
    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)
    from tree_1 import construct_table
    model_table = construct_table(model_ast, analyzed_data, module_dict)
    
    kwargs = {"in_channels":2, "out_channels": 3, "kernel_size":(2, 2)}    
    insert_multi_node(model_table, param_dict, new_layers = None, mutated_layer_indices=None, **kwargs)


if __name__ == '__main__':
    test_insert_multi_node()