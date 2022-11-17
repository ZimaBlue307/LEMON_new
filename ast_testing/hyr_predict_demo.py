import sys
import mindspore
import numpy as np
# from scripts.mutation.model_mutation_operators import *
# import train_ms.dataset as ds
# from scripts.tools.utils import ModelUtils
# from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation
import  ast
import inspect
import math
import astor
import astunparse
# from origin_model.ms_model.resnet20_cifar100_origin import MindSporeModel
from mindspore import Tensor
# from scripts.tools.utils import ToolUtils, ModelUtils
# from scripts.mutation.model_mutation_operators import GF_mut, WS_mut, NAI_mut
# from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation
import copy
from mindspore.nn import Dense
from mindspore.rewrite import *
from mindspore import nn, ops, Parameter
from alexnet_cifar10_origin.alexnet_cifar10_origin import MindSporeModel

class nodeInfo(object):

    '''
    node: operator, node
    name: operator name, string
    input: node
    output: node
    '''

    def __init__(self, node, name, input=None, output=None):
        self.node = node
        self.name = name
        self.input = input
        self.output = output

    def add_input(self, input):
        self.input = input

    def add_output(self, output):
        self.output = output

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
        self.is_return = False

    def set_return(self, value): #判断是不是最终的输出张量
        self.is_return = value

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

    def __init__(self, nodeList):
        super(ast.NodeVisitor, self).__init__()
        self.valueList = set()
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
                new_node_info = nodeInfo(op_node_info.node, op_node_info.name, input=arg_set, output=target_set)
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
        self.nodeList = set()

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            tmp = nodeInfo(node.value, node.targets[0].attr)
            self.nodeList.add(tmp)
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
    param_dict = mindspore.load_checkpoint(f'alexnet_cifar10_origin/alexnet_cifar10_origin.ckpt')
    mindspore.load_param_into_net(network, param_dict)
    input1=Tensor(np.random.randn(2,3,227,227),dtype=mindspore.float32)
    network_ast = astor.code_to_ast.parse_file(f'alexnet_cifar10_origin/alexnet_cifar10_origin.py')
    ast_string = astunparse.dump(network_ast)
    construct_function, init_function = None, None
    for item in network_ast.body:
        if isinstance(item, ast.ClassDef) and item.name == 'MindSporeModel':
            for ele in item.body:
                if isinstance(ele, ast.FunctionDef):
                    if ele.name == 'construct':
                        construct_function = ele # 找到construct函数
                    elif ele.name == '__init__':
                        init_function = ele # 找到init函数
    #get cell direction for each assign statement in construct function
    #get the api call from cell from assign statement in __init__ funtion
    init_visitor = initVisitor()
    init_visitor.visit(init_function)
    init_ast2code = ""
    for item in init_visitor.nodeList: #item是nodeinfo
        # self.node = node
        # self.name = name
        # self.input = input
        # self.output = output
        init_op_string = astor.to_source(item.node)
        # print(init_op_string)
        # print(item.name)
        init_op_name = item.name
        sub_init_string = "self.{} = {}\n".format(init_op_name, init_op_string)
        init_ast2code = init_ast2code + sub_init_string
    print(init_ast2code)
    
    
    print("lalalalallalalala")
    construct_visitor = constructVisitor(init_visitor.nodeList)
    construct_visitor.visit(construct_function) 
        # self.valueList = set() #保存输入
        # self.value_name = set() #保存输入张量的名称
        # self.nodeList = nodeList #保存init里的左边命名
        # self.graph = set() #图结构
        # self.input = None # 整个网络的输入
        
    construct_ast2code = ""
    for item in construct_visitor.graph: 
        # self.node = node
        # self.name = name
        # self.input = input
        # self.output = output #set. set中每个元素是个Value类型。其中包括：name, shape，以及是否是最终的输出张量的判断
        assign_output_name = item.output
        for ele in assign_output_name:
            out_Tensor_name = ele.name
        
        
            
        # opstring = astor.to_source(item.node, indent_with=' ' * 4, add_line_information=False, source_generator_class=astor.SourceGenerator)
        # print(opstring)
        # print(item.node)# init里右边的op定义
        # print(item.name) # construct里右边op的命名，init左边的命名
        # print(item.input)
        # print(item.output)
        # print("=================")
        # print(item.node) 是一个nodeinfo的object，是init里右边的op定义；
    # get_shape(construct_visitor.valueList, construct_visitor.graph, construct_visitor.input, network, input1)
    # for item in init_visitor.nodeList:
    #     print(type(item))
    #     print(item.node)
    #     print(item.name)
    #     print(item.input)
    #     print(item.output)
    #     print("=================")