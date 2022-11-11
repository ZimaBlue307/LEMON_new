import sys

import mindspore
import numpy as np
from scripts.mutation.model_mutation_operators import *

import keras
from scripts.tools.utils import ModelUtils
from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation


from origin_model.ms_model.resnet20_cifar100_origin import MindSporeModel
from mindspore import Tensor
from scripts.tools.utils import ToolUtils, ModelUtils
from scripts.mutation.model_mutation_operators import GF_mut, WS_mut, NAI_mut
from scripts.mutation.model_mutation_generators import generate_model_by_model_mutation
import copy
from mindspore.nn import Dense
from mindspore.rewrite import *
from mindspore import nn, Parameter

import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P

class Senior_Net(nn.Cell):
    def __init__(self, input_dims, output_dims):
        super(Net, self).__init__()
        self.matmul = P.MatMul()

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

class LeNet5(nn.Cell):
    """
    LeNet-5网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 卷积层，输入的通道数为num_channel,输出的通道数为6,卷积核大小为5*5
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # 卷积层，输入的通道数为6，输出的通道数为16,卷积核大小为5*5
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        # 全连接层，输入个数为16*5*5，输出个数为120
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        # 全连接层，输入个数为120，输出个数为84
        self.fc2 = nn.Dense(120, 84)
        # 全连接层，输入个数为84，分类的个数为num_class
        self.fc3 = nn.Dense(84, num_class)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 池化层
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # 多维数组展平为一维数组
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class No_Activation(nn.Cell):
    def __init__(self):
        super(No_Activation, self).__init__()
    def construct(self, x):
        results = x
        return results


if __name__ == "__main__":
    # lenet = LeNet5()
    # lenet_tree = mindspore.rewrite.SymbolTree.create(lenet)
    # len_layers, mapping = ToolUtils.get_layers(lenet)
    # no_activation = No_Activation()
    # no_activation_node = mindspore.rewrite.Node.create_call_cell(No_Activation(), targets = ['test_no_act_out'], name = 'no_act_node', is_sub_net = True)
    # # leakyrelu = mindspore.nn.LeakyReLU(alpha=0.2)
    # #leakyrelu_node = mindspore.rewrite.Node.create_call_cell(leakyrelu, targets = ['test_leakyrelu'])
    # # no_activation_tree = mindspore.rewrite.SymbolTree.create(no_activation)
    #
    # replacenode_list = [] #要替换进SymbolTree的节点列表
    # # noact_src_code = no_activation_tree.get_code()
    # # print(noact_src_code)
    # print("----------------------------------")
    #
    # for node in lenet_tree.nodes():
    #     print("the name of the node: ", node.get_name())
    #     # node_inputs = node.get_inputs()
    #     # if len(node_inputs) == 1:
    #     #     print("the input of this node: ", node_inputs[0].get_name())
    # print("----------------------------------")
    # i = 0
    # remove_index = 3
    # replace_index = 6
    # for node in lenet_tree.nodes():
    #     i = i+1
    #     #print("the name of the node: ", node.get_name())
    #     # if i == remove_index:
    #     #     #删除一个节点
    #     #     node_inputs = node.get_inputs()#获取当前节点的输入节点列表
    #     #     node_outputs = node.get_users()#获取当前节点的输出节点列表
    #     #     if len(node_inputs) == 1:
    #     #         node_input = node_inputs[0]
    #     #     if len(node_outputs) == 1:
    #     #         node_output = node_outputs[0]
    #     #     node_output.set_arg_by_node(0, node_input)#将另一个节点设置为当前节点的输入
    #     #     lenet_tree.erase_node(node)
    #
    #     if i == replace_index:
    #         #替换一个节点为其他激活函数层,例如no_activation；
    #         replacenode_list.append(no_activation_node)
    #         node_inputs = node.get_inputs()#获取当前节点的输入节点列表
    #         node_outputs = node.get_users()#获取当前节点的输出节点列表
    #         if len(node_inputs) == 1:
    #             node_input = node_inputs[0]
    #         if len(node_outputs) == 1:
    #             node_output = node_outputs[0]
    #         #position = lenet_tree.after(node)
    #         lenet_tree.replace(node, replacenode_list)#此时node的输出节点的输入节点已经修改为了新节点
    #         replacenode_list[0].set_arg_by_node(-1, node_input)#新节点的输入节点设为旧节点的输入节点
    #         #lenet_tree.erase_node(node)
    #
    # for node in lenet_tree.nodes():
    #     print("the name of the node: ", node.get_name())
    #     node_inputs = node.get_inputs()
    #     if len(node_inputs) == 1:
    #         print("the input of this node: ", node_inputs[0].get_name())
    #     if len(node_inputs) == 0:
    #         print("this node has no input node.")




    cifar100_dir = "dataset/cifar100/cifar-100-binary"
    dataset = mindspore.dataset.Cifar100Dataset(dataset_dir=cifar100_dir, usage='test', num_samples=128,
                                                shuffle=True)
    # for i, data in enumerate(dataset.create_dict_iterator()):
    #     label_tensor = data['fine_label']
    label_tensor = dataset.project(['fine_label'])
    print(label_tensor)
