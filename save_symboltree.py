import os.path

from origin_model.ms_model.resnet20_cifar100.resnet20_cifar100_origin import MindSporeModel

import mindspore
from mindspore import rewrite
from mindvision.classification.dataset import Cifar100

from origin_model.alexnet_cifar100_origin import MindSporeModelOpt
from mindspore import Tensor
import numpy as np
from mindspore.communication import init
from mindspore.train import Model
import json
import pickle


from scripts.mutation.model_shape_utils import Shape_uitls
# def insert_demo(tree, operator, )


if __name__ == "__main__":
    # mindspore.set_context(mode=mindspore.GRAPH_MODE, save_graphs=True, save_graphs_path='./data')
    # init()
    # mindspore.set_auto_parallel_context(full_batch=True, parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL)

    # save directory and model name
    save_root = 'lemon_outputs/resnet_cifar100'
    model_name = 'resnet_cifar100.py'
    ckpt_name = 'resnet20-cifar100_origin.ckpt'

    # first copy the model as the origin model
    origin_name = model_name.split('.')[0] + '_oringin.py'
    origin_path = os.path.join(save_root, origin_name)

    # load seed model and parameter
    network = MindSporeModel()
    param_dict = mindspore.load_checkpoint(f'origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt')
    mindspore.load_param_into_net(network, param_dict)
    network_tree = rewrite.SymbolTree.create(network)
    global_vars = network_tree._symbol_tree._global_vars
    # print(network_tree.dump())




    # # test shape name with the symbolTree
    # for node in enumerate(network_tree.nodes()):
    #     print(node)
    #
    #
    #
    #
    #
    #
    #
    # # save model
    # new_network = network_tree.get_network()
    # new_network_str = network_tree.get_code()
    # with open(origin_path, 'w') as f:
    #     f.write(new_network_str)
    # # network_tree.set_saved_file_name(origin_path)
    # # network_tree.save_network_to_file()
    #
    # # save ckpt
    # ckpt_path = os.path.join(save_root, ckpt_name)
    # mindspore.save_checkpoint(save_obj=new_network, ckpt_file_name=ckpt_path)
    # # save global vars
    # print(type(network_tree._symbol_tree._global_vars))
    # global_vars_name = origin_name.split('.')[0] + '_global.json'
    # global_vars_path = os.path.join(save_root, global_vars_name)

    # strs = pickle.dumps(network_tree._symbol_tree._global_vars)
    # print(strs)
    # unpickled_vars = pickle.loads(strs)
    # print(type(unpickled_vars))
    # with open(global_vars_path, 'wb') as f:
    #     pickle.dump(obj=network_tree._symbol_tree._global_vars, file=f)
    #
    # with open(global_vars_path, 'rb') as f:
    #     global_vars = pickle.load(f)
    # new_network = MindSporeModelOpt(network_tree._symbol_tree._global_vars)
    # # print(new_network)
    # print(type(network_tree._symbol_tree._global_vars))
    # new_network_tree = rewrite.SymbolTree.create(new_network)
    # new_network_tree.set_saved_file_name("resnet_cifar100_copy.py")
    # new_network_tree.save_network_to_file()

    # # print(train_x[:2].shape)
    # sample_x = train_x[0].reshape((-1, train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    # print(sample_x.shape)
    # ly_num, ly_map = utils.ToolUtils.get_layers(network)
    #
    # shape_list = list()
    # input_shape = sample_x.shape
    # output = Tensor(sample_x, dtype=mindspore.float32)
    # # output = network.construct(output)
    # output = mindspore.ops.transpose(output, (0, 3, 1, 2))


    # print(output.shape)
    # #send sample_x into the first cell
    # for i in range(len(ly_map.keys())):
    #     index = ly_map[i]
    #     layer_name, layer = utils.ModelUtils.get_layer(network, i)
    #     print(layer_name)
    #     output = layer.construct(output)
    #     print(output.shape)
    #     # input_shape.append({'input':input_shape, 'output':output.shape})

    # network_tree.set_saved_file_name("alexnet_cifar100_origin.py")
    # network_tree.save_network_to_file()
    # model = network_tree._symbol_tree._origin_network
