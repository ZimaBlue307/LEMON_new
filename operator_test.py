import sys

import mindspore
import numpy as np
from scripts.mutation.model_mutation_operators import *
import train_ms.dataset as ds

from scripts.tools.utils import ModelUtils
from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation
import  ast
import inspect
import math
import astor

# from origin_model.ms_model.resnet20_cifar100_origin import MindSporeModel
from mindspore import Tensor
from scripts.tools.utils import ToolUtils, ModelUtils
from scripts.mutation.model_mutation_operators import GF_mut, WS_mut, NAI_mut
from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation
import copy
from mindspore.nn import Dense
from mindspore.rewrite import *
from mindspore import nn, ops, Parameter
from origin_model.ms_model.resnet20_cifar100.resnet20_cifar100_origin import MindSporeModel

class Net(nn.Cell):
    def __init__(self, input_dims, output_dims):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

        self.weight_1 = Parameter(Tensor(np.random.randn(input_dims, 128), dtype=mindspore.float32), name='weight_1')
        self.bias_1 = Parameter(Tensor(np.zeros(128), dtype=mindspore.float32), name='bias_1')
        self.weight_2 = Parameter(Tensor(np.random.randn(128, 64), dtype=mindspore.float32), name='weight_2')
        self.bias_2 = Parameter(Tensor(np.zeros(64), dtype=mindspore.float32), name='bias_2')
        self.weight_3 = Parameter(Tensor(np.random.randn(64, output_dims), dtype=mindspore.float32), name='weight_3')
        self.bias_3 = Parameter(Tensor(np.zeros(output_dims), dtype=mindspore.float32), name='bias_3')

    def construct(self, x):
        x = self.matmul(x, self.weight_1)+self.bias_1
        x = self.matmul(x, self.weight_2)+self.bias_2
        x = self.matmul(x, self.weight_3)+self.bias_3
        return x

class nodeInfo(object):

    '''
    node is a representation of operators, it contains:
    input shape
    output shape
    cell function or operator function
    '''

    def __init__(self, node, name, input=None, output=None):
        self.node = node
        self.name = name
        self.input = input
        self.output = output
        self.sub_list = None

    def add_input(self, input):
        self.input = input

    def add_output(self, output):
        self.output = output

    # manage sub node list is current node is a module
    def add_sub_node(self, sub_node):
        if self.sub_list is None:
            self.sub_list = dict()
        self.sub_list[sub_node.name] = sub_node



class Value(object):
    '''
    Value contains the value to and from the node
    now contains
    value name: same as the name in the construct function
    input shape
    output shape
    '''

    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape
        self.value = None
        self.is_return = False

    def set_return(self, return_flag):
        self.is_return = return_flag
    def set_value(self, value):
        self.value = value

def is_in_list(node_name, lists):
    for item in lists:
        if item.name == node_name:
            return True, item
    return False, Node


class constructVisitor(ast.NodeVisitor):
    '''
    construct valueList
    connect node with value to construct the graph
    '''

    def __init__(self, nodeList, paraList):
        super(ast.NodeVisitor, self).__init__()
        self.valueList = paraList
        self.value_name = set()
        self.nodeList = nodeList
        self.graph = set()
        self.input = None #save the input value

    def is_in_valueList(self, valueNode):
        if valueNode.name in self.value_name:
            return True
        else:
            return False

    def visit_Assign(self, node):
        # visit assign node, save node into nodelist, save direction into node2node
        # print('find assign {}'.format(node))
        targets = node.targets
        value = node.value
        #get Value from value(Call) node
        #consider: value is a call node, or is a number. former to save in valuelist, later do not count
        target_set = set()
        for target in targets:
            tmp = Value(target.id)
            target_set.add(tmp)
            if not self.is_in_valueList(tmp):
                self.value_name.add(tmp.name)
                self.valueList.add(tmp)

        if isinstance(value, ast.Call):
            arg_set = set()
            for arg in value.args:
                tmp = Value(arg.id)
                arg_set.add(tmp)
                if not self.is_in_valueList(tmp):
                    self.value_name.add(tmp.name)

                    self.valueList.add(tmp)

            cur_op = value.func.attr
            #find node
            flag, op_node_info = is_in_list(cur_op, self.nodeList)
            if flag:
                new_node_info = nodeInfo(value, op_node_info.name, input=arg_set, output=target_set)
                self.graph.add(new_node_info)
            else:
                print("not find operator with current op name: {}".format(cur_op))

        self.generic_visit(node)
        return node


    def visit_FunctionDef(self, node):
        if node.name == 'construct':
            for i in range(len(node.args.args)):
                if i == 0:
                    continue
                tmp = Value(node.args.args[i].arg)
                self.value_name.add(tmp.name)
                self.valueList.add(tmp)
                self.input = tmp
        self.generic_visit(node)
        return node

    def visit_Return(self, node):

        self.generic_visit(node)
        return node



class initVisitor(ast.NodeVisitor):
    '''
    get node list according to init function
    '''
    def __init__(self):
        super(ast.NodeVisitor, self).__init__()
        self.nodeList = dict()
        self.para_list = dict()

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            # save parameter as value node
            if isinstance(node.value.func, ast.Name) and 'Parameter' in node.value.func.id:
                #get shape from parameter
                #how to get shape?
                #save param into the list
                parameter = Value(name=node.targets[0].attr)
                if not parameter.name in self.para_list.keys():
                    self.para_list[parameter.name] = parameter
            # if the assign is a attribute, means there is a operator
            # save operator info
            # for module, first save module info, then iter the module
            else:
                tmp = nodeInfo(node.value, node.targets[0].attr)
                if not tmp.name in self.nodeList.keys():
                    self.nodeList[tmp.name] = tmp
                    # if the operator is module
                    if 'module' in tmp.name.lower():
                        # iter add sub mode into node list.
                        pass
        self.generic_visit(node)
        return node


def get_name(sets):
    res = []
    for item in sets:
        res.append(item.name)
    return res

#support single input
def get_shape(valueList, graph, input, network, input_tensor):
    # find the node whose input contains 'input'
    cur_input_info_stack = []
    cur_input_info_stack.append(input)
    cur_input = input_tensor
    cur_output_info = None
    cur_output = None
    op = None
    while not len(cur_input_info_stack) == 0:
        cur_input_info = cur_input_info_stack[0]
        print(cur_input_info.name)
        for node in graph:
            flag, _ = is_in_list(input.name, node.input)
            if flag:
                op = node.name
                cur_output_info = node.output
        if op is None:
            print("can not find input! break")
            return None
        # if op not in network.cells, means the shape not changed
        cells = network._cells
        print('current operator {}'.format(op))
        flag = False
        if op in cells.keys():
            target_cell = cells[op]
            cur_output = target_cell.construct(cur_input)
            for item in cur_output_info:
                item.shape = cur_output.shape
        else:
            for item in cur_output_info:
                item.shape = cur_output.shape
        cur_input_info_stack = cur_input_info_stack[1:]
        for value in cur_output_info:
            cur_input_info_stack.append(value)
        cur_input = cur_output
        print(cur_input.shape)





if __name__ == "__main__":
    #get model
    network = MindSporeModel()
    param_dict = mindspore.load_checkpoint(f'origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt')
    mindspore.load_param_into_net(network, param_dict)
    input1=Tensor(np.random.randn(2,3,227,227),dtype=mindspore.float32)


    # network = Net(input_dims=3, output_dims=3)
    network_ast = astor.code_to_ast.parse_file(f'origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py')
    ast_str = astor.dump_tree(network_ast)
    # print(ast_str)
    construct_function, init_function = None, None
    for item in network_ast.body:
        if isinstance(item, ast.ClassDef) and item.name == 'MindSporeModel':
            # print(item.body)

            for ele in item.body:
                if isinstance(ele, ast.FunctionDef):
                    if ele.name == 'construct':
                        construct_function = ele
                    elif ele.name == '__init__':
                        init_function = ele
    #get cell direction for each assign statement in construct function
    #get the api call from cell from assign statement in __init__ funtion
    init_visitor = initVisitor()
    init_visitor.visit(init_function)
    for item in init_visitor.nodeList:
        print(item.name)
    construct_visitor = constructVisitor(init_visitor.nodeList)
    construct_visitor.visit(construct_function)
    print(construct_visitor.value_name)
    for item in construct_visitor.graph:
        print(item.name, get_name(item.input), get_name(item.output))

    get_shape(construct_visitor.valueList, construct_visitor.graph, construct_visitor.input, network, input1)


    #use ast to analyze construct function


    # network = Net(input_dims=20, output_dims=10)

    # indicies = ModelUtils.weighted_layer_indices(network)
    #
    # new_model = WS_mut(model=network,mutation_ratio=0.3, mutated_layer_indices=indicies)


    # for i, layer in enumerate(network.cells_and_names()):
    #     if i == 0:
    #         continue
    #     layer_name.append(layer[0])
    #     print(layer[1])
    #
    # params = []
    # # for param in network.trainable_params():
    # #     # print(param.shape)
    # #     params.append(param)
    # for param in network.get_parameters():
    #     params.append(param)
    #     print(param)
    #     print(param.T)
    #
    # print(len(layer_name))
    # print(len(params))





    #
    # # test keras layer input shape
    # model_path = 'origin_model/alexnet-cifar10_origin.h5'
    # origin_model = keras.models.load_model(model_path, custom_objects=ModelUtils.custom_objects())
    # origin_model.summary()
    # # mutated_model = generate_model_by_model_mutation(origin_model, 'LC')