import mindspore
import numpy as np
from scripts.mutation.model_mutation_operators import *
from mindspore import Tensor, Parameter
from scripts.tools.utils import ToolUtils
from mindspore.rewrite import *
from mindspore import nn
import mindspore.ops as P
from hyr_tmp_modify import *

class Module0_0(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features, conv2d_2_in_channels, conv2d_2_out_channels, conv2d_2_stride,
                 conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0_0, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999974752427e-07,
                                            momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(1, 1),
                                  stride=conv2d_2_stride,
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        return opt_relu_5

class Module5_0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module5_0, self).__init__()
        self.module0_0 = Module0_0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module0_1 = Module0_0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        module0_1_opt = self.module0_1(opt_conv2d_0)
        return module0_1_opt

class MindSporeModel_0(nn.Cell):

    def __init__(self):
        super(MindSporeModel_0, self).__init__()
        self.transpose_0 = P.Transpose()
        self.conv2d_8 = nn.Conv2d(in_channels=16,
                                  out_channels=64,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module5_0 = Module5_0(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=16,
                                 conv2d_2_out_channels=64,
                                 module0_0_batchnorm2d_0_num_features=64,
                                 module0_0_conv2d_2_in_channels=64,
                                 module0_0_conv2d_2_out_channels=16,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=16,
                                 module0_0_conv2d_4_out_channels=16,
                                 module0_1_batchnorm2d_0_num_features=64,
                                 module0_1_conv2d_2_in_channels=64,
                                 module0_1_conv2d_2_out_channels=16,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=16,
                                 module0_1_conv2d_4_out_channels=16)
        self.conv2d_26 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module5_1 = Module5_0(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=128,
                                 module0_0_batchnorm2d_0_num_features=128,
                                 module0_0_conv2d_2_in_channels=128,
                                 module0_0_conv2d_2_out_channels=64,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=64,
                                 module0_0_conv2d_4_out_channels=64,
                                 module0_1_batchnorm2d_0_num_features=128,
                                 module0_1_conv2d_2_in_channels=128,
                                 module0_1_conv2d_2_out_channels=64,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=64,
                                 module0_1_conv2d_4_out_channels=64)
        self.conv2d_51 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
    def construct(self, input_1):
        opt_transpose_0 = self.transpose_0(input_1, (0, 3, 1, 2))
        opt_conv2d_8 = self.conv2d_8(opt_transpose_0)
        module5_0_opt = self.module5_0(opt_conv2d_8) #test line. ignore this line of comment
        opt_conv2d_26 = self.conv2d_26(module5_0_opt)
        module5_1_opt = self.module5_1(opt_conv2d_26)
        return module5_1_opt

class LeNet5_1(nn.Cell):
    """
    LeNet-5网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5_1, self).__init__()
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

class hyr_relu1(nn.Cell):
    def __init__(self):
        super(hyr_relu1, self).__init__()
        self.reluhyr = nn.ReLU()
    def construct(self, x):
        x = self.reluhyr(x)
        return x

class No_Activation1(nn.Cell):
    def __init__(self):
        super(No_Activation1, self).__init__()
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

    print("----------check modified tree--------------")
    modified_lenet_tree = lenet_tree.get_code()
    print(modified_lenet_tree)

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
    hyr_relu1_obj = hyr_relu1() 
    for node in lenet_tree.nodes():
        i = i + 1
        if i == insert_index: #在node之后插入no_activation_node
            insert_position = lenet_tree.after(node)
            index_node_out_nodes = node.get_users()#插入节点的输出节点是前一个节点的输出节点 
            old_node = node
        if i == insert_index + 1:
            node_args = node.get_args() #相当于获取node的输入张量
            insert_node = mindspore.rewrite.Node.create_call_cell(cell = hyr_relu1_obj, targets = ["hyr_relu1_obj_output"], args  = node_args, name = "hyr_relu1_obj", is_sub_net = True)
            lenet_tree.insert(insert_position, insert_node)
            insert_node.set_arg_by_node(arg_idx = 0, src_node = old_node)            
            #插入节点的输入节点是当前节点
            node.set_arg_by_node(arg_idx = 0, src_node = insert_node) #这一行要先insert才不报错.把新节点作为下一个节点的输入；
    
    print("----------check modified tree--------------")
    modified_lenet_code = lenet_tree.get_code()
    print(modified_lenet_code)
    modified_model_path = f"./hyr_tmp_modify.py"
    with open(modified_model_path, "w") as fw:
        fw.write(modified_lenet_code)
    global_vars = lenet_tree._symbol_tree._global_vars
    
# done  
def nested_insertlayer(model_object, insert_layer):
    """
    assume changing: module5_0_opt = self.module5_0(opt_add_9).
    we add a layer inside module0_0 in module5_0.
    we have to make sure that only module0_0 in module5_0 is changed 
    while other module5_x and module0_x remain the same
    """
    model_tree = mindspore.rewrite.SymbolTree.create(model_object)
    model_old = model_tree.get_code()
    print(model_old)
    for i, node1 in enumerate(model_tree.nodes()):
        # 4:  module5
        if i == 4:
            sub_tree_5 = mindspore.rewrite.TreeNodeHelper.get_sub_tree(node1)
            for j , node2 in enumerate(sub_tree_5.nodes()):
                # 0:  input_x
                # 1:  module0
                # 2:  conv2d
                # 3:  module0_1
                # 4:  return
                if j == 1: # module0
                    sub_tree_0 = mindspore.rewrite.TreeNodeHelper.get_sub_tree(node2)
                    # sub_tree_0_code = sub_tree_0.get_code()
                    # print(sub_tree_0_code)
                    for k, node3 in enumerate(sub_tree_0.nodes()):
                        # 2 :  relu insert one layer in here
                        insert_index = 2 # relu
                        if k == insert_index:
                            insert_position = sub_tree_0.after(node3)
                            old_node = node3
                        if k == insert_index + 1:
                            node3_args = node3.get_args()
                            insert_node = mindspore.rewrite.Node.create_call_cell(cell = insert_layer, targets = ["hyr_obj_output"], args  = node3_args, name = "hyr_obj", is_sub_net = True)
                            sub_tree_0.insert(insert_position, insert_node)
                            insert_node.set_arg_by_node(arg_idx = 0, src_node = old_node)
                            node3.set_arg_by_node(arg_idx = 0, src_node = insert_node) 
    print("-------------------------------------------")
    print("----------check modified tree--------------")
    print("-------------------------------------------")
    
    modified_tree = model_tree.get_code()
    print(modified_tree)
    
def get_global_var(model_object):
    model_tree = mindspore.rewrite.SymbolTree.create(model_object)
    model_code = model_tree.get_code()
    # print(model_code)
    global_vars = model_tree._symbol_tree._global_vars
    model_path = f"./hyr_tmp_modify.py"
    with open(model_path, "w") as fw:
        fw.write(model_code)
        
    new_model = MindSporeModel(global_vars)
    new_model_tree = mindspore.rewrite.SymbolTree.create(new_model)
    model_code2 = new_model_tree.get_code()
    print(model_code2)        
    

if __name__ == "__main__":
    # resnet20_cifar100_test = MindSporeModel_0()
    # hyr_relu1_obj = hyr_relu1()
    # get_global_var(resnet20_cifar100_test)
    lenet = LeNet5_1()
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
    hyr_relu1_obj = hyr_relu1() 
    for node in lenet_tree.nodes():
        i = i + 1
        if i == insert_index: #在node之后插入no_activation_node
            insert_position = lenet_tree.after(node)
            index_node_out_nodes = node.get_users()#插入节点的输出节点是前一个节点的输出节点 
            old_node = node
        if i == insert_index + 1:
            node_args = node.get_args() #相当于获取node的输入张量
            insert_node = mindspore.rewrite.Node.create_call_cell(cell = hyr_relu1_obj, targets = ["hyr_relu1_obj_output"], args  = node_args, name = "hyr_relu1_obj", is_sub_net = True)
            lenet_tree.insert(insert_position, insert_node)
            insert_node.set_arg_by_node(arg_idx = 0, src_node = old_node)            
            #插入节点的输入节点是当前节点
            node.set_arg_by_node(arg_idx = 0, src_node = insert_node) #这一行要先insert才不报错.把新节点作为下一个节点的输入；
    
    print("----------check modified tree--------------")
    modified_lenet_code = lenet_tree.get_code()
    print(modified_lenet_code)
    modified_model_path = f"./hyr_tmp_modify.py"
    with open(modified_model_path, "w") as fw:
        fw.write(modified_lenet_code)
    global_vars = lenet_tree._symbol_tree._global_vars
    from hyr_tmp_modify import *
    new_model = LeNet5_1(global_vars)