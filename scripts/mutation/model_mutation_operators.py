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
import random
import os
import warnings
from scripts.logger.lemon_logger import Logger
import pickle
import datetime
import ast
import astunparse

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

mylogger = Logger()

#add function to determine whether the layer is an activation function
def is_layer_in_activation_list(operator_name):
    import mindspore
    activation_list = [mindspore.nn.layer.activation.Softmin, mindspore.nn.layer.activation.Softmax, mindspore.nn.layer.activation.LogSoftmax,
                       mindspore.nn.layer.activation.ReLU, mindspore.nn.layer.activation.ReLU6, mindspore.nn.layer.activation.RReLU,
                       mindspore.nn.layer.activation.SeLU, mindspore.nn.layer.activation.SiLU, mindspore.nn.layer.activation.Tanh,
                       mindspore.nn.layer.activation.Tanhshrink, mindspore.nn.layer.activation.Hardtanh, mindspore.nn.layer.activation.GELU,
                       mindspore.nn.layer.activation.FastGelu, mindspore.nn.layer.activation.Sigmoid, mindspore.nn.layer.activation.Softsign,
                       mindspore.nn.layer.activation.PReLU, mindspore.nn.layer.activation.LeakyReLU, mindspore.nn.layer.activation.HSigmoid, 
                       mindspore.nn.layer.activation.HSwish, mindspore.nn.layer.activation.ELU, mindspore.nn.layer.activation.LogSigmoid,
                       mindspore.nn.layer.activation.LRN, mindspore.nn.layer.activation.SoftShrink, mindspore.nn.layer.activation.HShrink,
                       mindspore.nn.layer.activation.CELU, mindspore.nn.layer.activation.Threshold, mindspore.nn.layer.activation.Mish
                       ]
    for i in activation_list:
        if isinstance(layer, i):
            return True
    return False

def is_layer_softmax(layer):
    import mindspore
    softmax_list = [mindspore.nn.layer.activation.Softmax, mindspore.nn.layer.activation.LogSoftmax]
    for i in softmax_list:
        if isinstance(layer, i):
            return True
    return False


activation_list_without_softmax = [mindspore.nn.layer.activation.Softmin, mindspore.nn.layer.activation.LogSoftmax,
                       mindspore.nn.layer.activation.ReLU, mindspore.nn.layer.activation.ReLU6, mindspore.nn.layer.activation.RReLU,
                       mindspore.nn.layer.activation.SeLU, mindspore.nn.layer.activation.SiLU, mindspore.nn.layer.activation.Tanh,
                       mindspore.nn.layer.activation.Tanhshrink, mindspore.nn.layer.activation.Hardtanh, mindspore.nn.layer.activation.GELU,
                       mindspore.nn.layer.activation.FastGelu, mindspore.nn.layer.activation.Sigmoid, mindspore.nn.layer.activation.Softsign,
                       mindspore.nn.layer.activation.PReLU, mindspore.nn.layer.activation.LeakyReLU, mindspore.nn.layer.activation.HSigmoid, 
                       mindspore.nn.layer.activation.HSwish, mindspore.nn.layer.activation.ELU, mindspore.nn.layer.activation.LogSigmoid,
                       mindspore.nn.layer.activation.LRN, mindspore.nn.layer.activation.SoftShrink, mindspore.nn.layer.activation.HShrink,
                       mindspore.nn.layer.activation.CELU, mindspore.nn.layer.activation.Threshold, mindspore.nn.layer.activation.Mish
                       ]
#add function to determine whether the layer is an activation function except softmax
def is_layer_in_activation_list_without_softmax(layer):
    import mindspore
    for i in activation_list_without_softmax:
        if isinstance(layer, i):
            return True
    return False


#done
def _assert_indices(mutated_layer_indices: List[int] , depth_layer: int):#done

    assert max(mutated_layer_indices) < depth_layer,"Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0,"Min index should be greater than or equal to zero"

# may need to change
def _shuffle_conv1d(layer, mutate_ratio):
    new_layer = copy.deepcopy(layer)
    parameters = new_layer.get_parameters()
    for i, parameter in enumerate(parameters):
        val = parameter.data.asnumpy()
        if len(val.shape) > 1:
            val_shape = val.shape
            num_of_output_channels, num_of_input_channels, filter_width, filter_height = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list, (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
                copy_list = np.reshape(copy_list,(filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:,output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,(num_of_output_channels, num_of_input_channels, filter_width, filter_height))
            from mindspore import Tensor
        val = Tensor(val, dtype=mindspore.float32)
        parameter.set_data(val)
    return new_layer
#done
def _shuffle_conv2d(layer, mutate_ratio):
    new_layer = copy.deepcopy(layer)
    parameters = new_layer.get_parameters()
    # new_weights = []
    for i, parameter in enumerate(parameters):
        # val is bias if len(val.shape) == 1
        # if len(val.shape) > 1:
        val = parameter.data.asnumpy()
        if len(val.shape) > 1:
            val_shape = val.shape
            num_of_output_channels, num_of_input_channels, filter_width, filter_height = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list, (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
                copy_list = np.reshape(copy_list,(filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:,output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,(num_of_output_channels, num_of_input_channels, filter_width, filter_height))
            from mindspore import Tensor
        val = Tensor(val, dtype=mindspore.float32)
        parameter.set_data(val)
            # val_shape = val.shape
            # filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
            # mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            # for output_channel in mutate_output_channels:
            #     copy_list = val.copy()
            #     copy_list = np.reshape(copy_list,(filter_width * filter_height * num_of_input_channels, num_of_output_channels))
            #     selected_list = copy_list[:,output_channel] #what does this mean?——have nothing to do with shape_format
            #     shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
            #     copy_list[:, output_channel] = shuffle_selected_list
            #     val = np.reshape(copy_list,(filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        # new_weights.append(val)
    return new_layer


# this might be used when improving the mutation function within a layer
def _shuffle_conv3d(layer, mutate_ratio):
    new_layer = copy.deepcopy(layer)
    parameters = new_layer.get_parameters()
    # new_weights = []
    for i, parameter in enumerate(parameters):
        # val is bias if len(val.shape) == 1
        # if len(val.shape) > 1:
        val = parameter.data.asnumpy()
        if len(val.shape) > 1:
            val_shape = val.shape
            num_of_output_channels, num_of_input_channels, filter_width, filter_height = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list, (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
                copy_list = np.reshape(copy_list,(filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:,output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,(num_of_output_channels, num_of_input_channels, filter_width, filter_height))
            from mindspore import Tensor
        val = Tensor(val, dtype=mindspore.float32)
        parameter.set_data(val)
    return new_layer

#done
def _shuffle_dense(layer, mutate_ratio):
    new_weights = []
    new_layer = copy.deepcopy(layer)
    parameters = new_layer.get_parameters()
    # for val in weights:
    for i, parameter in enumerate(parameters):
        # val is bias if len(val.shape) == 1
        val = parameter.data.asnumpy()
        if len(val.shape) > 1:
            val_shape = val.shape
            output_dim, input_dim = val_shape
            mutate_output_dims = utils.ModelUtils.generate_permutation(output_dim, mutate_ratio)
            copy_list = val.copy()
            copy_list = np.reshape(copy_list, (input_dim, output_dim))
            for output_dim in mutate_output_dims:
                selected_list = copy_list[:, output_dim]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_dim] = shuffle_selected_list
            copy_list = np.reshape(output_dim, input_dim)
            val = copy_list
        # new_weights .append(val)
        from mindspore import Tensor
        val = Tensor(val, dtype=mindspore.float32)
        parameter.set_data(val)
    return new_layer


def _LA_model_scan(irtable, model, new_layers, mutated_layer_indices = None):
    from lemon_tree.mutation_utils_mindspore import LayerUtils
    layer_utils = LayerUtils()
    nodeList = irtable.nodeList
    positions_to_add = np.arange(len(nodeList) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(nodeList))
    insertion_points = {}
    available_new_layers = [layer for layer in
                            layer_utils.available_model_level_layers.keys()] if new_layers is None else new_layers
    for node_index in nodeList.keys():
        node = nodeList[node_index]
        operator_name = node.operator_name
        # input_list = node.input_list
        # output_list = node.output_list
        output_shape = node.shape
        if 'softmax' in operator_name.lower():
            break
        if node_index in positions_to_add:
            for available_new_layer in available_new_layers:
                # first judge whether the input is legal
                if layer_utils.is_input_legal[available_new_layer](output_shape):
                    if node_index not in insertion_points.keys():
                        insertion_points[node_index] = [available_new_layer]
                    else:
                        insertion_points[node_index].append(available_new_layer)
    return insertion_points


def _MLA_model_scan(irtable, model, new_layers, mutated_layer_indices=None):
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


def _LC_and_LR_scan(irtable):
    available_layer_indices = []
    nodeList = irtable.nodeList
    for node_index in nodeList.keys():
        node = nodeList[node_index]
        operator_name = node.operator_name
        input_list = node.input_list
        output_list = node.output_list
        output_shape = node.shape
        #先判断是不是激活函数，如果是则continue，如果不是则继续；
        if 'softmax' in operator_name.lower():
            break
        #保证只有一个输入和输出，并且不是第一层和最后一层
        if len(input_list) == 1 and len(output_list) == 1 and input_list[0] >=0 and output_list[0] >= 0:
            input_node = nodeList[input_list[0]]
            input_shape = input_node.shape
            if output_shape == input_shape:
                available_layer_indices.append(node_index)
            else:
                continue
    np.random.shuffle(available_layer_indices)
    return available_layer_indices


def _LS_scan(model, irtable):
    shape_dict = {}
    nodeList = irtable.nodeList
    for node_index in nodeList.keys():
        node = nodeList[node_index]
        #如果该层是softmax，则break；暂时先使用这个判断；
        lower_op_name = node.operator_name.lower()
        if 'softmax' in lower_op_name:
            break
        input_list = node.input_list
        if len(input_list) > 1:
            continue
        output_shape = node.shape
        input_node = nodeList[input_list[0]]
        input_shape = input_node.shape
        input_shape_connect = "-".join(input_shape)
        output_shape_connect = "-".join(output_shape)
        k = "+".join([input_shape_connect, output_shape_connect])
        if k not in shape_dict.keys():
            shape_dict[k] = [node_index]
        else:
            shape_dict[k].append(node_index)
    return shape_dict

#done and tested
def GF_mut(model, mutation_ratio, distribution='normal', STD=0.1, lower_bound=None, upper_bound=None):

    valid_distributions = ['normal', 'uniform']
    assert distribution in valid_distributions, 'Distribution %s is not support.' % distribution
    if distribution == 'uniform' and (lower_bound is None or upper_bound is None):
        mylogger.error('Lower bound and Upper bound is required for uniform distribution.')
        raise ValueError('Lower bound and Upper bound is required for uniform distribution.')

    # mylogger.info('copying model...')

    GF_model = utils.ModelUtils.model_copy(model, 'GF')#need to change
    # mylogger.info('model copied')

    # layers = GF_model.cells_and_names()
    len_layers, _ = utils.ToolUtils.get_layers(model)
    
    chosed_index = np.random.randint(0, len_layers)
    # chosed_index = 13
    # layer = GF_model.layers[chosed_index] #change

    #use iteration to get layer
    layer_name, layer = utils.ModelUtils.get_layer(GF_model, chosed_index)
    mylogger.info('executing mutation in {} index {}'.format(layer_name, chosed_index))
    # weights = layer.get_weights()#change
    parameters = layer.get_parameters()
    from mindspore.ops import Reshape
    from mindspore import Tensor
    for i, parameter in enumerate(parameters):
        weight = parameter.data
        weight_shape = weight.shape
        # param_num = 1
        # for i in range(len(weight_shape)):
        #     param_num *= weight_shape[i]
        weight = weight.asnumpy()
        weight_flat = weight.flatten()
        permu_num = math.floor(len(weight_flat) * mutation_ratio)
        permutation = np.random.permutation(len(weight_flat))[:permu_num]
        STD = math.sqrt(weight_flat.var()) * STD
        weight_flat[permutation] += np.random.normal(scale=STD, size=len(permutation))
        weight_flat = weight_flat.reshape(weight_shape)
        new_weight = Tensor(weight_flat, dtype=mindspore.float32)
        parameter.set_data(new_weight)

    # new_weights = []
    # for weight in weights:
    #     weight_shape = weight.shape
    #     weight_flat = weight.flatten()
    #     permu_num = math.floor(len(weight_flat) * mutation_ratio)
    #     permutation = np.random.permutation(len(weight_flat))[:permu_num]
    #     STD = math.sqrt(weight_flat.var()) * STD
    #     weight_flat[permutation] += np.random.normal(scale=STD, size=len(permutation))
    #     weight = weight_flat.reshape(weight_shape)
    #     new_weights.append(weight)
    # layer.set_weights(new_weights)

    return GF_model

#done, but can be updated
def WS_mut(model, mutation_ratio, mutated_layer_indices=None):
    WS_model = utils.ModelUtils.model_copy(model, 'WS')
    # layers = WS_model.layers
    layers = WS_model.cells_and_names()
    # depth_layer = len(layers)
    depth_layer, layer_map = utils.ToolUtils.get_layers(WS_model)
    mutated_layer_indices = np.arange(depth_layer) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, depth_layer)
        np.random.shuffle(mutated_layer_indices)
        i = mutated_layer_indices[0]
        layer_name, layer = utils.ModelUtils.get_layer(model, i)
        # weights = layer.get_weights()
        weights = list(layer.get_parameters())
        # layer_name = type(layer).__name__
        layer_name = layer.cls_name
        # WS_layer = copy.deepcopy(layer)
        # if layer_name == "Conv2d" and len(weights) != 0:
        if "Conv2d" in layer_name and len(weights) != 0:
            # layer.set_weights(_shuffle_conv2d(weights, mutation_ratio))
            layer = _shuffle_conv2d(layer, mutation_ratio)
        elif "Conv1d" in layer_name and len(weights) != 0:
            # layer = _shuffle_conv1d(layer, mutation_ratio)
            pass
        elif "Conv3d" in layer_name and len(weights) != 0:
            # layer = _shuffle_conv3d(layer, mutation_ratio)
            pass
        elif layer_name == "Dense" and len(weights) != 0:
            layer.set_weights(_shuffle_dense(weights, mutation_ratio))
            # layer = WS_layer
        elif layer_name == "BatchNormalization" and len(weights) != 0:
            pass
        elif layer_name == "DepthwiseConv2D" and len(weights) != 0:
            pass
        else:
            pass
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return WS_model

#done
def NEB_mut(model, mutation_ratio, mutated_layer_indices=None):
    NEB_model = utils.ModelUtils.model_copy(model, 'NEB')
    # layers = NEB_model.layers
    ly_num, ly_map = utils.ToolUtils.get_layers(NEB_model)
    mutated_layer_indices = np.arange(ly_num - 1) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, ly_num)
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        for i in mutated_layer_indices:
            layer_name, layer = utils.ModelUtils.get_layer(NEB_model, i)
            # layer = layers[i]
            # skip if layer is not in white list
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue

            # weights = layer.get_weights()
            weights = list(layer.get_parameters())
            if len(weights) > 0:
                if isinstance(weights, list):
                    # assert len(weights) == 2
                    if len(weights) != 2:
                        continue
                    else:
                        w, b = weights
                        weights_w = w.asnumpy()
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        weights_w[permutation] = np.zeros(weights_w[0].shape)
                        weights_w = weights_w.transpose()
                        weights_b = b.asnumpy()
                        weights_b[permutation] = 0
                        from mindspore import Tensor
                        weights_w = Tensor(weights_w, dtype=mindspore.float32)
                        weights_b = Tensor(weights_b, dtype=mindspore.float32)
                        w.set_data(weights_w)
                        b.set_data(weights_b)
                else:
                    assert isinstance(weights, np.ndarray)
                    weights_w = weights
                    weights_w = weights_w.transpose()
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] = np.zeros(weights_w[0].shape)
                    weights_w = weights_w.transpose()
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
        return NEB_model
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")

#why some weights not in list form?
#done
def NAI_mut(model, mutation_ratio, mutated_layer_indices=None):
    NAI_model = utils.ModelUtils.model_copy(model, 'NAI')
    #layers = NAI_model.cells_and_names()
    ly_num, ly_map = utils.ToolUtils.get_layers(NAI_model)

    mutated_layer_indices = np.arange(ly_num - 1) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, ly_num)
        np.random.shuffle(mutated_layer_indices)
        layer_utils = LayerUtils()
        for i in mutated_layer_indices:
            layer_name, layer = utils.ModelUtils.get_layer(model, i)
            # new_layer = copy.deepcopy(layer)
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue
            weights = list(layer.get_parameters())
            if len(weights) > 0:
                if isinstance(weights, list):
                    if len(weights) != 2:
                        continue
                    else:
                        w, b = weights
                        weights_w = w.asnumpy()
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        # print(permutation)
                        weights_w[permutation] *= -1
                        weights_w = weights_w.transpose()
                        weights_b = b.asnumpy()
                        weights_b[permutation] *= -1
                        weights = weights_w, weights_b
                        from mindspore import Tensor
                        weights_w = Tensor(weights_w, dtype=mindspore.float32)
                        weights_b = Tensor(weights_b, dtype=mindspore.float32)
                        w.set_data(weights_w)
                        b.set_data(weights_b)
                else:
                    weights_w = weights[0]
                    weights_w = weights_w.transpose()
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    # print(permutation)
                    weights_w[permutation] *= -1
                    weights_w = weights_w.transpose()
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return NAI_model

#done
def NS_mut(model, mutated_layer_indices=None):
    NS_model = utils.ModelUtils.model_copy(model, 'NS')
    # layers = NS_model.layers
    layers = NS_model.cells_and_names()
    ly_num, ly_map = utils.ToolUtils.get_layers(NS_model)
    mutated_layer_indices = np.arange(ly_num - 1) if mutated_layer_indices is None else mutated_layer_indices
    # _assert_indices(mutated_layer_indices, len(layers))
    layer_utils = LayerUtils()
    for i in mutated_layer_indices:
        # layer = layers[i]
        layer_name, layer = utils.ModelUtils.get_layer(NS_model, i)
        if not layer_utils.is_layer_in_weight_change_white_list(layer):
            continue
        # weights = layer.get_weights()
        weights = list(layer.get_parameters())
        if len(weights) > 0:
            if isinstance(weights, list):
                if len(weights) != 2:
                    continue
                w, b = weights
                # weights_w, weights_b = weights
                weights_w = w.asnumpy()
                weights_w = weights_w.transpose()
                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]

                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()

                    weights_b = b.asnumpy()
                    weights_b[permutation[0]], weights_b[permutation[1]] = \
                        weights_b[permutation[1]].copy(), weights_b[permutation[0]].copy()

                    from mindspore import Tensor
                    weights_w = Tensor(weights_w, dtype=mindspore.float32)
                    weights_b = Tensor(weights_b, dtype=mindspore.float32)
                    w.set_data(weights_w)
                    b.set_data(weights_b)

                    # layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            else:
                assert isinstance(weights, np.ndarray)
                weights_w = weights
                weights_w = weights_w.transpose()
                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]

                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()
                    weights = [weights_w]

                    layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            break

    return NS_model

#in lemon_raw, this function removes the activation function of a layer.
#but in ms, this function should be understood as replacing an activation layer 
# with no_activation. assuming that all act layers only have 1 input and output
# done
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
    # print(type(global_vars))
    # print(global_vars.keys())
    # print("===========================")
    # print(global_vars['handler'])
    from tmp.test_ARem import MindSporeModel
    ARem_new_model = MindSporeModel(global_vars)
    return ARem_new_model

# replaces the activation function of a layer 
# with a randomly selected activation function
# def ARep_mut(model, new_activations=None, mutated_layer_indices=None):
#     activation_utils = ActivationUtils()
#     ARep_model = utils.ModelUtils.model_copy(model, 'ARep')
#     # len_layers = utils.ToolUtils.get_layers(ARep_model)[0]
#     ARep_tree = mindspore.rewrite.SymbolTree.create(ARep_model)
#     len_ARep_tree = 0#该模型symboltree的长度
#     mapping_index_node = dict()#key是数字索引，value是node
#     mapping_node_parent = dict()#key是数字索引，value是数字索引对应node的parent_tree
#     len_ARep_tree, mapping_index_node, mapping_node_parent = utils.ToolUtils.judge_node(ARep_tree, len_ARep_tree, mapping_index_node, mapping_node_parent)
#     # the activation of last layer should not be replaced
#     mutated_layer_indices = np.arange(len_ARep_tree - 1) if mutated_layer_indices is None else mutated_layer_indices
#     np.random.shuffle(mutated_layer_indices)
#     _assert_indices(mutated_layer_indices, len_ARep_tree)
#     for i in mutated_layer_indices:
#         #先获得要修改的ARep_node是哪个节点；
#         ARep_node = mapping_index_node[i]
#         ARep_node_instance = ARep_node.get_instance()
#         if is_layer_in_activation_list_without_softmax(ARep_node_instance):
#             # print("the node need to remove: ", ARem_node.get_name())
#             parent_tree = mapping_node_parent[i]#获得这个节点所在的symboltree
#             #print(ARem_node.get_name(), "has parent_tree: ", parent_tree)
#             ARem_node_inputs = ARep_node.get_inputs()#获取当前节点的输入节点列表
#             ARem_node_outputs = ARep_node.get_users()#获取当前节点的输出节点列表
#             if len(ARem_node_inputs) == 1 and len(ARem_node_outputs) == 1:
#                 #print("the node ", ARem_node.get_name(), "has only one input.")
#                 ARem_node_input = ARem_node_inputs[0]
#                 ARem_node_output = ARem_node_outputs[0]
#                 # need to add the logic of repalcing a layer;
#                 break
#             else:
#                 if len(ARem_node_inputs) > 1:
#                     print(ARep_node.get_name(), " has multiple inputs.")
#                     continue
#                 if len(ARem_node_outputs) > 1:
#                     print(ARep_node.get_name(), " has multiple outputs.")
#                     continue
#     # 还要补上return model的办法；
#     # ARem_tree.set_saved_file_name("./tmp/test_ARem.py")
#     # ARem_tree.save_network_to_file()
#     # global_vars = ARem_tree._symbol_tree._global_vars
#     # return ARem_new_model
    

# Layer Addition: selects a layer, whose input shape and
# output shape are consistent and then inserts it to a 
# compatible position in the model.
def LA_mut(model, new_layers=None, mutated_layer_indices=None):
    layer_utils = LayerUtils()#need to change
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_utils.available_model_level_layers.keys():
                mylogger.error('Layer {} is not supported.'.format(layer))
                raise Exception('Layer {} is not supported.'.format(layer))
    LA_model = utils.ModelUtils.model_copy(model, 'LA')

    insertion_points = _LA_model_scan(LA_model, new_layers, mutated_layer_indices)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
            .format(insertion_points[key], key, type(model.layers[key])))
    layers_index_avaliable = list(insertion_points.keys())
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('insert {} after {}'.format(layer_name_to_insert, LA_model.layers[layer_index_to_insert].name))
    
    # insert new layer
    if model.__class__.__name__ == 'Sequential':
        import mindspore
        new_model = mindspore.nn.SequentialCell()
        for i, layer in enumerate(LA_model.layers):
            new_layer = LayerUtils.clone(layer)
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output_shape
                new_model.add(layer_utils.available_model_level_layers[layer_name_to_insert](output_shape))
    else:

        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output_shape
            new_layer = layer_utils.available_model_level_layers[layer_name_to_insert](output_shape)
            x = new_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LA_model, operation={LA_model.layers[layer_index_to_insert].name: layer_addition})

    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    import time
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        if layer_name.endswith('_copy_LA'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))

    # import keras.backend as K
    # K.batch_set_value(tuples)
    #batch_set_value,一次设置多个张量变量的值。
    #tuples：元组 (tensor, value) 的列表。 value 应该是一个 Numpy 数组。
    import mindspore
    import numpy as np
    for i in tuples[1]:
        tuples[0] = mindspore.Tensor(tuples[1], mindspore.float32)
    #not for sure.tuples[1]是一个numpy数组，还是多个numpy数组呢？
        
    return new_model

#compared with LA, the input and output don't need to be consistent
def MLA_mut(model, new_layers = None, mutated_layer_indices=None):
    # mutiple layers addition
    layer_matching = LayerMatching()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    MLA_model = utils.ModelUtils.model_copy(model, 'MLA')
    insertion_points = _MLA_model_scan(model, new_layers, mutated_layer_indices)
    mylogger.info(insertion_points)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                             .format(insertion_points[key], key, type(model.layers[key])))

    # use logic: randomly select a new layer available to insert into the layer which can be inserted
    layers_index_avaliable = list(insertion_points.keys())
    # layer_index_to_insert = np.max([i for i in insertion_points.keys()])
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('choose to insert {} after {}'.format(layer_name_to_insert, MLA_model.layers[layer_index_to_insert].name))
    # insert new layers
    if model.__class__.__name__ == 'Sequential':
        import mindspore
        new_model = mindspore.nn.SequentialCell()
        
        for i, layer in enumerate(MLA_model.layers):
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += "_copy"
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output.shape.as_list()
                layers_to_insert = layer_matching.layer_concats[layer_name_to_insert](output_shape)
                for layer_to_insert in layers_to_insert:
                    layer_to_insert.name += "_insert"
                    mylogger.info(layer_to_insert)
                    new_model.add(layer_to_insert)
        new_model.build(MLA_model.input_shape)
    else:
        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output.shape.as_list()
            new_layers = layer_matching.layer_concats[layer_name_to_insert](output_shape)
            for l in new_layers:
                l.name += "_insert"
                mylogger.info('insert layer {}'.format(str(l)))
                x = l(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(MLA_model, operation={MLA_model.layers[layer_index_to_insert].name: layer_addition})

    tuples = []
    import time
    start_time = time.time()

    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_MLA'):
            key = layer_name[:-9]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    end_time = time.time()
    print('set weight cost {}'.format(end_time - start_time))

    return new_model

#layer Copy: copies a layer, whose input shape and out-put 
#shape are consistent, and then inserts the copied layer
#to concatenate the original layer
def LC_mut(model, mutated_layer_indices=None):
    LC_model = utils.ModelUtils.model_copy(model, 'LC')
    mapping_index_node = dict()#key是数字索引，value是node
    mapping_node_parent = dict()#key是数字索引，value是数字索引对应node的parent_tree
    #scan先暂时放着
    available_layer_indices = _LC_and_LR_scan(LC_model, mutated_layer_indices, mapping_index_node, mapping_node_parent)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate node to copy (input and output shape should be same)')
        return None

    # use logic: copy the layer with last index in available_layer_indices
    copy_layer_index = available_layer_indices[-1]
    LC_node = mapping_index_node[copy_layer_index]
    # copy_layer_name = LC_model.layers[copy_layer_index].name + '_repeat' node名字暂时不修改

    mylogger.info('choose to copy layer {}'.format(LC_node.get_name()))
    
    parent_tree = mapping_node_parent[copy_layer_index]
    LC_node_inputs = LC_node.get_inputs()
    LC_node_outputs = LC_node.get_users()
    LC_node_input = LC_node_inputs[0]
    LC_node_output = LC_node_outputs[0]
    LC_position = parent_tree.after(LC_node)
    # parent_tree.insert(LC_position, )

    
    # update weights unchanged
    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LC'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        if layer_name + '_copy_LC_repeat' == copy_layer_name:
            for sw, w in zip(new_model_layers[copy_layer_name].weights, layer_weights):
                shape_sw = np.shape(sw)
                shape_w = np.shape(w)
                assert len(shape_sw) == len(shape_w)
                assert shape_sw[0] == shape_w[0]
                tuples.append((sw, w))

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model

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

#Layer Removal: removes a layer, whose input shape and output shape are consistent
def LR_mut(model_path, mutated_layer_indices=None):
    # LR_model = utils.ModelUtils.model_copy(model, 'LR') #model copy function, jump for now
    model_dir = os.path.dirname(model_path)
    model_name = model_path.split("/")[-1]
    table_name = tuple(model_name.split("."))[0] + '_table.pkl'
    with open(table_name, 'r') as f:
        irtable = pickle.load(os.path.join(model_dir, table_name))
    # scan函数先放着；
    available_layer_indices = _LC_and_LR_scan(irtable)
    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate node to remove (input and output shape should be same)')
        return None
    # use logic: remove the layer with last index in available_layer_indices
    remove_layer_index = available_layer_indices[-1]
    LR_node = mapping_index_node[remove_layer_index]
    mylogger.info('choose to remove node {}'.format(LR_node.get_name()))
    #这里没有修改node的名称，在lemon_raw里有修改；
    #这里应该不需要区分sequential和非sequential了
    parent_tree = mapping_node_parent[remove_layer_index]
    LR_node_inputs = LR_node.get_inputs()
    LR_node_outputs = LR_node.get_users()
    LR_node_input = LR_node_inputs[0]
    LR_node_output = LR_node_outputs[0]
    LR_node_output.set_arg_by_node(0, LR_node_input)
    parent_tree.erase_node(LR_node)
    LR_tree.set_saved_file_name("./tmp/test_LR.py")
    LR_tree.save_network_to_file()
    LR_global_vars = LR_tree._symbol_tree._global_vars
    from tmp.test_LR import MindsporeModel
    LR_new_model = MindsporeModel(LR_global_vars)
    return LR_new_model

#Layer Switch: switches two layers, both of which have the same input shape and output shape
def LS_mut(model):
    LS_model = utils.ModelUtils.model_copy(model,"LS")
    shape_dict = _LS_scan(LS_model)
    layers = LS_model.layers

    swap_list = []
    for v in shape_dict.values():
        if len(v) > 1:
            swap_list.append(v)
    if len(swap_list) == 0:
        mylogger.warning("No layers to swap!")
        return None
    swap_list = swap_list[random.randint(0, len(swap_list)-1)]
    choose_index = random.sample(swap_list, 2)
    mylogger.info('choose to swap {} ({} - {}) and {} ({} - {})'.format(layers[choose_index[0]].name,
                                                                layers[choose_index[0]].input.shape,
                                                                layers[choose_index[0]].output.shape,
                                                                layers[choose_index[1]].name,
                                                                layers[choose_index[1]].input.shape,
                                                                layers[choose_index[1]].output.shape))
    if model.__class__.__name__ == 'Sequential':
        #import keras
        import mindspore
        new_model = mindspore.nn.SequentialCell()
        for i, layer in enumerate(layers):
            if i == choose_index[0]:
                new_model.add(LayerUtils.clone(layers[choose_index[1]]))
            elif i == choose_index[1]:
                new_model.add(LayerUtils.clone(layers[choose_index[0]]))
            else:
                new_model.add(LayerUtils.clone(layer))
    else:
        layer_1 = layers[choose_index[0]]
        layer_2 = layers[choose_index[1]]
        new_model = utils.ModelUtils.functional_model_operation(LS_model, {layer_1.name: lambda x, layer: LayerUtils.clone(layer_2)(x),
                                                           layer_2.name: lambda x, layer: LayerUtils.clone(layer_1)(x)})

    # update weights
    assert len(new_model.layers) == len(model.layers)
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LS'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


if __name__ == '__main__':
    pass
