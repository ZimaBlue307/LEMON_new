#assuming all the input_shapes are channel first;
import copy
from ctypes import util
import sys
from operator import truediv
from scripts.mutation.mutation_utils import No_Activation
# from mutation_utils import No_Activation
import mindspore
from mindspore.rewrite import *
import numpy as np
from scripts.tools import utils
import math
from typing import *
from scripts.mutation.mutation_utils import *
from scripts.mutation.layer_matching import LayerMatching
from scripts.mutation.model_mutation_generators import *
import random
import os
import warnings
from scripts.logger.lemon_logger import Logger
import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

mylogger = Logger()

def ARem_mut(model, mutated_layer_indices=None):
    ARem_model = utils.ModelUtils.model_copy(model, 'ARem')
    ARem_tree = mindspore.rewrite.SymbolTree.create(ARem_model)
    len_ARem_tree = 0#该模型symboltree的长度
    mapping_index_node = dict()#key是数字索引，value是node
    mapping_node_parent = dict()#key是数字索引，value是数字索引对应node的parent_tree
    len_ARem_tree, mapping_index_node, mapping_node_parent = utils.ToolUtils.judge_node(ARem_tree, len_ARem_tree, mapping_index_node, mapping_node_parent)    
    mutated_layer_indices = np.arange(len_ARem_tree-1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len_ARem_tree)

    for i in mutated_layer_indices:
        #先获得要修改的ARem_node是哪个节点；
        ARem_node = mapping_index_node[i]
        ARem_node_instance = ARem_node.get_instance()
        if is_layer_in_activation_list_without_softmax(ARem_node_instance):
            # print("the node need to remove: ", ARem_node.get_name())
            
            #print(ARem_node.get_name(), "has parent_tree: ", parent_tree)

            ARem_node_inputs = ARem_node.get_inputs()#获取当前节点的输入节点列表
            ARem_node_outputs = ARem_node.get_users()#获取当前节点的输出节点列表
            if len(ARem_node_inputs) == 1 and len(ARem_node_outputs) == 1:
                #print("the node ", ARem_node.get_name(), "has only one input.")
                parent_tree = mapping_node_parent[i]#获得这个节点所在的symboltree
                ARem_node_input = ARem_node_inputs[0]
                ARem_node_output = ARem_node_outputs[0]
                ARem_node_output.set_arg_by_node(0, ARem_node_input)
                parent_tree.erase_node(ARem_node)
                break
            else:
                if len(ARem_node_inputs) > 1:
                    print(ARem_node.get_name(), " has multiple inputs.")
                    continue
                if len(ARem_node_outputs) > 1:
                    print(ARem_node.get_name(), " has multiple outputs.")
                    continue
    # 还要补上return model的办法；
    ARem_tree.set_saved_file_name("./tmp/test_ARem.py")
    ARem_tree.save_network_to_file()
    # print(ARem_tree.get_code())
    global_vars = ARem_tree._symbol_tree._global_vars
    from tmp.test_ARem import MindSporeModel
    ARem_new_model = MindSporeModel(global_vars)
    return ARem_new_model


if __name__ == '__main__':
    from origin_model.ms_model.resnet20_cifar100.resnet20_cifar100_origin import MindSporeModel
    model = MindSporeModel()
    mapping_index_node = dict()
    mapping_node_parent = dict()
    new_model = generate_model_by_model_mutation(model, operator = 'ARem')
    