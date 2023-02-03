import mindspore
import numpy as np
from scripts.mutation.model_mutation_operators import *
from mindspore import Tensor
from scripts.tools.utils import ToolUtils
from scripts.mutation.model_mutation_operators import GF_mut, WS_mut, NAI_mut
from mindspore.rewrite import *
from mindspore import nn

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

class hyr_relu(nn.Cell):
    def __init__(self):
        super(hyr_relu, self).__init__()
        self.reluhyr = nn.ReLU()
    def construct(self, x):
        x = self.reluhyr(x)
        return x

class No_Activation(nn.Cell):
    def __init__(self):
        super(No_Activation, self).__init__()
    def construct(self, x):
        results = x
        return results

#done
def remove_func():
    lenet = LeNet5()
    lenet_tree = mindspore.rewrite.SymbolTree.create(lenet)
    len_layers, mapping = ToolUtils.get_layers(lenet)

    for node in lenet_tree.nodes():
        print("the name of the node: ", node.get_name())
        print()
    print("----------------------------------")
    i = 0
    remove_index = 3# relu
    insert_index = 6# relu_1
    replace_index = 5 # conv2d
    for node in lenet_tree.nodes():
        i = i + 1
        if i == remove_index: #OK 删除暂时没有bug
            remove_node_inputs = node.get_inputs()
            remove_node_outputs = node.get_users()
            if len(remove_node_inputs) == 1:
                remove_node_input = remove_node_inputs[0]
            if len(remove_node_outputs) == 1:
                remove_node_output = remove_node_outputs[0]
            remove_node_output.set_arg_by_node(0, remove_node_input)
            lenet_tree.erase_node(node)
            break
    modified_lenet_tree = lenet_tree.get_network()
    print(modified_lenet_tree)
    print("----------check modified tree--------------")
    for node in lenet_tree.nodes():
        node_inputs = node.get_inputs()
        node_outputs = node.get_users()

        if len(node_inputs) == 1:
            print("the INput of this node: ", node_inputs[0].get_name())
        if len(node_inputs) == 0 or node_inputs is None:
            print("this node has no INput node.")

        print("the name of the node: ", node.get_name())
        
        if len(node_outputs) == 1:
            print("the OUTput of this node: ", node_outputs[0].get_name())
        if len(node_outputs) == 0 or node_outputs is None:
            print("this node has no OUTput node.")

        print("当前节点的参数：", node.get_args())
        print("当前节点的所有属性: ", node.get_attributes())
        print("获取当前节点带key的参数: ", node.get_kwargs())
        print("--------------------------------")

def replacelayer_func():
    lenet = LeNet5()
    lenet_tree = mindspore.rewrite.SymbolTree.create(lenet)
    for node in lenet_tree.nodes():
        print("the name of the node: ", node.get_name())
    print("----------------------------------")
    i = 0
    remove_index = 3# relu
    insert_index = 6# relu_1
    replace_index = 5 # conv2d
    for node in lenet_tree.nodes():
        i = i + 1
        if i == replace_index: #no_activation_node替换当前节点
            replacenode_list = [] #要替换进SymbolTree的节点列表
            no_activation = No_Activation()
            no_activation_node = mindspore.rewrite.Node.create_call_cell(No_Activation(), targets = ['test_no_act_out'], name = 'no_act_node', is_sub_net = True)
            old_node = node
            new_node = no_activation_node
            test1 = old_node._node._inputs
            replacenode_list.append(new_node)
            replace_node_inputs = node.get_inputs()
            replace_node_outputs = node.get_users()
            if len(replace_node_inputs) == 1:
                replace_node_input = replace_node_inputs[0]
            if len(replace_node_outputs) == 1:
                replace_node_output = replace_node_outputs[0]
            lenet_tree.replace(old_node, replacenode_list)
            replace_node_output.set_arg_by_node(0, no_activation_node)
            # 上面两行按照这个顺序写是不会报错的
            # no_activation_node.set_arg_by_node(0, replace_node_input)
            #这么写，替换的node可以和后面的node接上，但是不能和前面的node接上；

    modified_lenet_tree = lenet_tree.get_network()
    print(modified_lenet_tree)

    print("----------check modified tree--------------")
    for node in lenet_tree.nodes():
        node_inputs = node.get_inputs()
        node_outputs = node.get_users()

        if len(node_inputs) == 1:
            print("the INput of this node: ", node_inputs[0].get_name())
        if len(node_inputs) == 0 or node_inputs is None:
            print("this node has no INput node.")

        print("the name of the node: ", node.get_name())
        
        if len(node_outputs) == 1:
            print("the OUTput of this node: ", node_outputs[0].get_name())
        if len(node_outputs) == 0 or node_outputs is None:
            print("this node has no OUTput node.")

        print("当前节点的参数：", node.get_args())
        print("当前节点的所有属性: ", node.get_attributes())
        print("获取当前节点带key的参数: ", node.get_kwargs())
        print("--------------------------------")

# done
def insertlayer_func():
    lenet = LeNet5()
    lenet_tree = mindspore.rewrite.SymbolTree.create(lenet)
    # for node in lenet_tree.nodes():
    #     print("the name of the node: ", node.get_name())
    modified_lenet_tree_old = lenet_tree.get_code()
    print(modified_lenet_tree_old)
    print("----------------------------------")
    # lenet_tree.print_node_tabulate(lenet_tree)
    i = 0
    remove_index = 3# relu
    insert_index = 6# relu_1
    replace_index = 5 # conv2d
    hyr_relu_obj = hyr_relu() 
    for node in lenet_tree.nodes():
        i = i + 1
        if i == insert_index: #在node之后插入no_activation_node
            insert_position = lenet_tree.after(node)
            index_node_out_nodes = node.get_users()#插入节点的输出节点是前一个节点的输出节点 
            old_node = node
        if i == insert_index + 1:
            node_args = node.get_args() #相当于获取node的输入张量
            insert_node = mindspore.rewrite.Node.create_call_cell(cell = hyr_relu_obj, targets = ["hyr_relu_obj_output"], args  = node_args, name = "hyr_relu_obj", is_sub_net = True)
            lenet_tree.insert(insert_position, insert_node)
            insert_node.set_arg_by_node(arg_idx = 0, src_node = old_node)            
            #插入节点的输入节点是当前节点
            node.set_arg_by_node(arg_idx = 0, src_node = insert_node) #这一行要先insert才不报错.把新节点作为下一个节点的输入；
    
    
    modified_lenet_tree = lenet_tree.get_code()
    print(modified_lenet_tree)
    lenet_tree.print_node_tabulate() #为什么用不了？


if __name__ == "__main__":
    insertlayer_func()
