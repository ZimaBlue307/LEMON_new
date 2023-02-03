import os
import pickle
import math
from PIL import Image
import warnings
import datetime
import configparser
import numpy as np
from scripts.logger.lemon_logger import Logger
from scripts.mutation.mutation_utils import LayerUtils
import inspect
import ast
import astunparse
import copy
import sys
# import astor
import json
import collections
import mindspore
from mindspore import Parameter, Tensor

np.random.seed(20200501)
warnings.filterwarnings("ignore")
"""Set seed and Init cuda"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

main_logger = Logger()

class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def model_copy(model, mode=''):
        """LEMON_RAW is to clone each layer one by one and add them 
        to the new model, so as to achieve the effect of 
        completely replicating the seed model. 
        (check clone function in scripts/mutation/mutation_utils.py)
        Now we directly copy the seed model, get the new model, 
        and then import the parameters into the new model
        
        get the network of the model;
        get the ckpt of the model;
        copy the network;
        load the ckpt to the network
        create a new model
        """

        import copy
        import mindspore
        from mindspore import load_checkpoint, load_param_into_net
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')
        mindspore.save_checkpoint(model, "./tmp/model.ckpt")
        new_model = copy.deepcopy(model)
        # param_dict = load_checkpoint("resnet50-2_32.ckpt")
        param_dict = load_checkpoint("./tmp/model.ckpt")
        load_param_into_net(new_model, param_dict)
        os.remove('./tmp/model.ckpt')
        return new_model
        # from scripts.mutation.mutation_utils import LayerUtils
        # import keras
        # suffix = '_copy_' + mode
        # if model.__class__.__name__ == 'Sequential':
        #     new_layers = []
        #     for layer in model.layers:
        #         new_layer = LayerUtils.clone(layer)
        #         new_layer.name += suffix
        #         new_layers.append(new_layer)
        #     new_model = keras.Sequential(layers=new_layers, name=model.name + suffix)
        # else:
        #     new_model = ModelUtils.functional_model_operation(model, suffix=suffix)

        # s = datetime.datetime.now()
        # new_model.set_weights(model.get_weights())
        # e1 = datetime.datetime.now()
        # td1 = e1 - s
        # h, m, s = ToolUtils.get_HH_mm_ss(td1)
        # print("Set model weights! {} hour,{} min,{} sec".format(h, m, s))
        # del model
        # return new_model
        

    @staticmethod
    def functional_model_operation(model, operation=None, suffix=None):
        from scripts.mutation.mutation_utils import LayerUtils
        input_layers = {}
        output_tensors = {}
        model_output = None
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    input_layers[layer_name].append(layer.name)

        output_tensors[model.layers[0].name] = model.input

        for layer in model.layers[1:]:
            layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]

            if operation is not None and layer.name in operation.keys():
                x = layer_input_tensors
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = operation[layer.name](x, cloned_layer)
            else:
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = cloned_layer(layer_input_tensors)

            output_tensors[layer.name] = x
            model_output = x

        import keras
        return keras.Model(inputs=model.inputs, outputs=model_output)

    @staticmethod
    def save_initial_weights(model):
        weights = model.get_weights()
        np.save('initial_weights.npy', weights)

    @staticmethod
    def load_initial_weights(model):
        weights = np.load('initial_weights.npy')
        model.set_weights(weights)
        return model

    @staticmethod
    def save_layers_output(path, layers_output):

        dirname = os.path.dirname(path)
        if len(dirname)>0 and (not os.path.exists(dirname)):
            os.makedirs(dirname)
        with open(path,'wb') as f:
            pickle.dump(layers_output,f)

    @staticmethod
    def load_layers_output(path):
        if not os.path.exists(path):
            return None
        with open(path,'rb') as f:
            layers_output = pickle.load(f)
        return layers_output

    @staticmethod
    def layer_divation(model, model_nodes, layer_index, layers_output_1, layers_output_2, epsilon=1e-7):
        layer = model.layers[layer_index]
        # get all of its input layers
        input_layers_index = []
        for node in layer._inbound_nodes:
            if node not in model_nodes:
                continue
            for l in node.inbound_layers:
                from keras.engine.input_layer import InputLayer
                if isinstance(l, InputLayer):
                    continue
                # find the index of l in model
                for i, model_layer in enumerate(model.layers):
                    if l == model_layer:
                        input_layers_index.append(i)
                        break
                else:
                    raise Exception('can not find the layer in model')
        # calculate the divation of current layer
        cur_output_1 = layers_output_1[layer_index]
        cur_output_2 = layers_output_2[layer_index]
        delta_cur = MetricsUtils.delta(cur_output_1, cur_output_2)[0] # the second value of delta is sum()

        if len(input_layers_index) == 0:
            delta_pre = 0
        else:
            delta_pre_list = []
            for i in input_layers_index:
                pre_output_1 = layers_output_1[i]
                pre_output_2 = layers_output_2[i]
                delta_pre_list.append(MetricsUtils.delta(pre_output_1, pre_output_2)[0])
            delta_pre = np.max(delta_pre_list, axis=0)
        return delta_cur, (delta_cur - delta_pre) / (delta_pre + epsilon), [model.layers[i].name for i in input_layers_index]

    @staticmethod
    def layers_divation(model, layers_output_1, layers_output_2):
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v
        layers_divation = []
        for i in range(len(model.layers)):
            layers_divation.append(ModelUtils.layer_divation(model, relevant_nodes, i, layers_output_1, layers_output_2))
        return layers_divation

    @staticmethod
    def layers_output(model, input):
        from keras import backend as K
        # print(K.backend()+" in loadmodel")
        from keras.engine.input_layer import InputLayer
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [l.output for l in
                                       (model.layers[1:]
                                        if isinstance(model.layers[0], InputLayer)
                                        else model.layers)])
        if isinstance(model.layers[0], InputLayer):
            layers_output = [input]
            layers_output.extend(get_layer_output([input, 0]))
        else:
            layers_output = get_layer_output([input, 0])
        return layers_output

    @staticmethod
    def layers_input(model, input):
        inputs = [[input]]
        from keras import backend as K
        from keras.engine.input_layer import InputLayer
        for i, layer in enumerate(model.layers):
            if i == 0:
                continue
            if i == 1 and isinstance(model.layers[0], InputLayer):
                continue
            get_layer_input = K.function([model.layers[0].input, K.learning_phase()],
                                         layer.input if isinstance(layer.input, list) else [layer.input])
            inputs.append(get_layer_input([input, 0]))
        return inputs

    @staticmethod
    #done
    def generate_permutation(size_of_permutation, extract_portion):
        assert extract_portion <= 1
        num_of_extraction = math.floor(size_of_permutation * extract_portion)
        permutation = np.random.permutation(size_of_permutation)
        permutation = permutation[:num_of_extraction]
        #permutation等于permutation的前num_of_extraction位；
        return permutation

    @staticmethod
    #done
    def shuffle(a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        #numpy.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组
        length = len(a)
        permutation = np.random.permutation(length)#随机生成一个length长度的array
        index_permutation = np.arange(length)#生成array[0,1,2..., length-1]
        shuffled_a[permutation] = a[index_permutation]
        return shuffled_a

    @staticmethod
    def compile_model(model, optimer, loss, metric:list):
        model.compile(optimizer=optimer,
                      loss=loss,
                      metrics=metric)
        return model

    @staticmethod
    def custom_objects():
        from scripts.mutation.mutation_utils import ActivationUtils
        objects = {}
        objects['no_activation'] = ActivationUtils.no_activation
        objects['leakyrelu'] = ActivationUtils.leakyrelu
        return objects

    @staticmethod
    def get_layer(model, index):
        '''
        return layer with the index
        :param model:
        :param index:
        :return:
        '''
        layers = model.cells_and_names()
        layer = next(layers)
        ly_num, ly_map = ToolUtils.get_layers(model)
        map_index = ly_map[index]
        for i in range(map_index):
            layer = next(layers)
        return layer[0], layer[1]


    @staticmethod
    def weighted_layer_indices(model):
        '''
        return layer indicies that have weights.
        noted that returned indices should use with layer_map, or we can use without layer_map
        :param model:
        :return:
        '''
        indices = []

        layers = model.cells_and_names()
        layer = next(layers)
        ly_num, ly_map = ToolUtils.get_layers(model)
        iter_index = 0
        for i in range(ly_num):
            ly_index = ly_map[i]
            for j in range(ly_index - iter_index):
                layer = next(layers)
            iter_index = ly_index
            parameters = list(layer[1].get_parameters())
            weight_shape = len(parameters)
            if weight_shape >= 1:
                indices.append(i)

        # for i, layer in enumerate(layers):
        #     weight_count = layer.count_params()
        #     if weight_count > 0:
        #         indices.append(i)
        return indices

    @staticmethod
    def is_valid_model(inputs_backends,backends_nums, threshold=0.95):
        invalid_status_num = 0
        inputs_values = list(inputs_backends.values())
        # results like (1500,1) is valid
        if inputs_values[0].shape[1] == 1:
            return True
        else:
            for inputs in inputs_backends.values():
                indice_map = {}
                for input in inputs:
                    max_indice = np.argmax(input)
                    if max_indice not in indice_map.keys():
                        indice_map[max_indice] = 1
                    else:
                        indice_map[max_indice] += 1
                for indice in indice_map.keys():
                    if indice_map[indice] > len(inputs) * threshold:
                        invalid_status_num += 1

            return False if invalid_status_num == backends_nums else True


class DataUtils:

    @staticmethod
    def image_resize(x, shape):
        x_return = []
        for x_test in x:
            tmp = np.copy(x_test)
            img = Image.fromarray(tmp.astype('uint8')).convert('RGB')
            img = img.resize(shape, Image.ANTIALIAS)
            x_return.append(np.array(img))
        return np.array(x_return)

    # @staticmethod
    # def get_data_by_exp(exp, test_size): #like exp = alexnet_cifar100
    #     """
    #     old: return x_test and y_test
    #     new: return dataset after certain operations
    #     """
    #     import mindspore
    #     # import keras
    #     # import keras.backend as K
    #     # K.set_image_data_format("channels_last")
    #     # K.set_image_data_format("channels_first")
    #     # in mindspore, data format is channel_first, so we may need to skip up three lines.

    #     lemon_cfg = configparser.ConfigParser()
    #     lemon_cfg.read("./config/ms_experiments.conf")
    #     # dataset_dir = lemon_cfg['parameters']['dataset_dir']
    #     # x_test = y_test = []
    #     # adding new elif branch
    #     if 'cifar100' in exp:
    #         dataset_name = "cifar100"
    #         cifar100_dir = "dataset/cifar100/cifar-100-binary"
    #         dataset = mindspore.dataset.Cifar100Dataset(dataset_dir = cifar100_dir, usage='test', num_samples = test_size, shuffle=True) 
    #         #In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label"(100) and "coarse_label"(20)
    #         # 可以用下述两行获得image和label;
    #         # for data in one_hot_dataset.create_dict_iterator():
    #         #     print(type(data["image"]))
    #         # x_test = dataset.image
    #         # y_test = dataset.label
    #         # x_test = DataUtils.get_cifar100_data(x_test)
    #         # def get_cifar100_data(x_test):
    #         #     x_test = x_test.astype('float32') / 255.0
    #         #     h, w = 32, 32
    #         #     x_test = x_test.reshape(x_test.shape[0], 3, h, w)
    #         # several lines next is used to implement fonction get_cifar100_data
    #         import mindspore.dataset.transforms as C
    #         import mindspore.dataset.vision as CV
    #         from mindspore.dataset.vision import Inter
    #         resize_height, resize_width = 32, 32
    #         rescale_param = 1.0 / 255.0
    #         shift_param = -1.0
    #         one_hot_opt = C.OneHot(num_classes=100) 
    #         rescale_op = CV.Rescale(rescale_param, shift_param)
    #         resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    #         dataset = dataset.map(operations = one_hot_opt, input_columns=["fine_label"]) #把细标签转换为独热编码
    #         dataset = dataset.map(operations = rescale_op, input_columns=["image"])
    #         dataset = dataset.map(operations = resize_op, input_columns=["image"])

    #         # get the shape of fine_label.
    #         # this variable is used when generating metrics result
    #         for i, data in enumerate(dataset.create_dict_iterator()):
    #             label_shape = data['fine_label'].shape
    #             break # since all shape are the same, so break after getting into the loop for the first time
        
        # elif 'fashion-mnist' in exp:
        #     _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        #     x_test = DataUtils.get_fashion_mnist_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # elif 'mnist' in exp:
        #     _, (x_test, y_test) = keras.datasets.mnist.load_data()
        #     x_test = DataUtils.get_mnist_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # elif 'cifar10' in exp:
        #     _, (x_test, y_test) = keras.datasets.cifar10.load_data()
        #     x_test = DataUtils.get_cifar10_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10) #把类别标签转换为独热编码
        # elif 'imagenet' in exp:
        #     input_precessor = DataUtils.imagenet_preprocess_dict()
        #     input_shapes_dict = DataUtils.imagenet_shape_dict()
        #     model_name = exp.split("-")[0]
        #     shape = input_shapes_dict[model_name]
        #     data_path = os.path.join(dataset_dir,"sampled_imagenet-1500.npz")
        #     data = np.load(data_path)
        #     x, y = data['x_test'], data['y_test']
        #     x_resize = DataUtils.image_resize(np.copy(x),shape)
        #     x_test = input_precessor[model_name](x_resize)
        #     y_test = keras.utils.to_categorical(y, num_classes=1000)
        # elif 'sinewave' in exp:
        #     """
        #     see more details in
        #     https://github.com/StevenZxy/CIS400/tree/f69489c0624157ae86b5d8ddb1fa99c89a927256/code/LSTM-Neural-Network-for-Time-Series-Prediction-master
        #     """
        #     import pandas as pd
        #     dataframe = pd.read_csv(f"{dataset_dir}/sinewave.csv")
        #     test_size,seq_len = 1500, 50
        #     data_test = dataframe.get("sinewave").values[-(test_size + 50):]
        #     data_windows = []
        #     for i in range(test_size):
        #         data_windows.append(data_test[i:i + seq_len])
        #     data_windows = np.array(data_windows).astype(float).reshape((test_size,seq_len,1))
        #     data_windows = np.array(data_windows).astype(float)
        #     x_test = data_windows[:, :-1]
        #     y_test = data_windows[:, -1, [0]]

        # elif 'price' in exp:
        #     """see more details in https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject"""
        #     x_test, y_test = DataUtils.get_price_data(dataset_dir)

        # TODO: Add your own data preprocessing here
        # Note: The returned inputs should be preprocessed and labels should decoded as one-hot vector which could be directly feed in model.
        # Both of them should be returned in batch, e.g. shape like (1500,28,28,1) and (1500,10)
        # elif 'xxx' in exp:
        #     x_test, y_test = get_your_data(dataset_dir)

        # return dataset, dataset_name

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    @staticmethod
    #一些operators在三个测试的mindspore版本中，api不一样。故本函数的参数多传入一个backend信息以作判断
    def get_data_by_exp_with_bk(exp, test_size, backend_name, cfg_name): #like exp = alexnet_cifar100
        """
        old: return x_test and y_test
        new: return dataset after certain operations
        """
        import mindspore
        # from keras.utils import to_categorical
        # import keras
        # import keras.backend as K
        # K.set_image_data_format("channels_last")
        # K.set_image_data_format("channels_first")
        # in mindspore, data format is channel_first, so we may need to skip up three lines.
        lemon_cfg = configparser.ConfigParser()
        # lemon_cfg.read("./config/experiments.conf")
        main_logger.info(f"cfg path {cfg_name}")
        cfg_name = os.path.join('config/', cfg_name)
        lemon_cfg.read(cfg_name)
        dataset_dir = lemon_cfg['parameters']['dataset_dir']
        x_test = y_test = []
        # adding new elif branch
        if 'cifar100' in exp:
            # In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label"(100) and "coarse_label"(20)
            dataset_name = "cifar100"
            cifar100_dir = os.path.join(dataset_dir, "cifar100/cifar-100-binary")
            dataset = mindspore.dataset.Cifar100Dataset(dataset_dir = cifar100_dir, usage='test', num_samples = test_size, shuffle=False) # batch_size=32, download=True,
            import mindspore.dataset.transforms as transforms
            import mindspore.dataset.vision as CV
            from mindspore.dataset.vision import Inter
            resize_height, resize_width = 32, 32
            rescale_param = 1.0 / 255.0
            shift_param = 0.0 #不平移
            if backend_name == "mindspore1.7.1":
                one_hot_opt = transforms.py_transforms.OneHotOp(num_classes=100)
                rescale_op = CV.c_transforms.Rescale(rescale_param, shift_param)
                resize_op = CV.py_transforms.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
                ndarray2PIL = CV.py_transforms.ToPIL()
                totensor_op = CV.py_transforms.ToTensor()
                from mindspore.dataset.transforms.py_transforms import Compose
                image_op_list = Compose([ndarray2PIL, resize_op, totensor_op])
                dataset = dataset.map(operations = one_hot_opt, input_columns=["fine_label"]) #把细标签转换为独热编码   
                dataset = dataset.map(operations = rescale_op, input_columns=["image"])
                
            elif backend_name == 'mindspore1.8.1':
                one_hot_opt = transforms.OneHot(num_classes=100) 
                rescale_op = CV.Rescale(rescale_param, shift_param)
                resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
                dataset = dataset.map(operations = one_hot_opt, input_columns=["fine_label"]) #把细标签转换为独热编码   
                dataset = dataset.map(operations = rescale_op, input_columns=["image"])
                dataset = dataset.map(operations = resize_op, input_columns=["image"])
            elif backend_name == 'mindspore1.6.2':
                one_hot_opt = transforms.py_transforms.OneHotOp(num_classes=100)
                dataset = dataset.map(operations=one_hot_opt, input_columns=["fine_label"])
                rescale_op = CV.c_transforms.Rescale(rescale_param, shift_param)
                dataset = dataset.map(operations=rescale_op, input_columns=["image"])
                
        elif 'cifar10' in exp:
            dataset_name = 'cifar10'
            cifar10_path = 'dataset/cifar10'
            cifar10_python_path = os.path.join(dataset_dir, 'cifar-10-batch-py', 'test_batch')
            # 使用python方法獲得y_true
            # cifar_python = DataUtils.unpickle(cifar10_python_path)
            # y_true = np.array(cifar_python['labels'][:test_size])
            # y_true = to_categorical(y_true, num_classes=10)
            # 使用ms方法獲得dataset
            dataset = mindspore.dataset.Cifar10Dataset(dataset_dir=cifar10_path, usage='test', num_samples=test_size,
                                                        shuffle=False)
            import mindspore.dataset.transforms as transforms
            import mindspore.dataset.vision as CV
            from mindspore.dataset.vision import Inter
            resize_height, resize_width = 32, 32
            rescale_param = 1.0 / 255.0
            shift_param = 0.0  # 不平移
            if backend_name == 'mindspore1.7.1':
                one_hot_opt = transforms.py_transforms.OneHotOp(num_classes=10)
                rescale_op = CV.c_transforms.Rescale(rescale_param, shift_param)
                # resize_op = CV.py_transforms.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
                # ndarray2PIL = CV.py_transforms.ToPIL()
                # totensor_op = CV.py_transforms.ToTensor()
                from mindspore.dataset.transforms.py_transforms import Compose
                # image_op_list = Compose([ndarray2PIL, resize_op, totensor_op])
                dataset = dataset.map(operations=one_hot_opt, input_columns=["label"])  # 把细标签转换为独热编码
                dataset = dataset.map(operations=rescale_op, input_columns=["image"])
            elif backend_name == 'mindspore1.8.1':
                one_hot_opt = transforms.OneHot(num_classes=10)
                rescale_op = CV.Rescale(rescale_param, shift_param)
                resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
                # In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label"(100) and "coarse_label"(20)
                dataset = dataset.map(operations=one_hot_opt, input_columns=["label"])  # 把细标签转换为独热编码
                dataset = dataset.map(operations=rescale_op, input_columns=["image"])
                dataset = dataset.map(operations=resize_op, input_columns=["image"])
            elif backend_name == 'mindspore1.6.2':
                one_hot_opt = transforms.py_transforms.OneHotOp(num_classes=10)
                dataset = dataset.map(operations=one_hot_opt, input_columns=["label"])
                rescale_op = CV.c_transforms.Rescale(rescale_param, shift_param)
                dataset = dataset.map(operations=rescale_op, input_columns=["image"])
                
        # elif 'fashion-mnist' in exp:
        #     _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        #     x_test = DataUtils.get_fashion_mnist_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # elif 'mnist' in exp:
        #     _, (x_test, y_test) = keras.datasets.mnist.load_data()
        #     x_test = DataUtils.get_mnist_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # elif 'cifar10' in exp:
        #     _, (x_test, y_test) = keras.datasets.cifar10.load_data()
        #     x_test = DataUtils.get_cifar10_data(x_test)
        #     y_test = keras.utils.to_categorical(y_test, num_classes=10) #把类别标签转换为独热编码
        # elif 'imagenet' in exp:
        #     input_precessor = DataUtils.imagenet_preprocess_dict()
        #     input_shapes_dict = DataUtils.imagenet_shape_dict()
        #     model_name = exp.split("-")[0]
        #     shape = input_shapes_dict[model_name]
        #     data_path = os.path.join(dataset_dir,"sampled_imagenet-1500.npz")
        #     data = np.load(data_path)
        #     x, y = data['x_test'], data['y_test']
        #     x_resize = DataUtils.image_resize(np.copy(x),shape)
        #     x_test = input_precessor[model_name](x_resize)
        #     y_test = keras.utils.to_categorical(y, num_classes=1000)
        # elif 'sinewave' in exp:
        #     """
        #     see more details in
        #     https://github.com/StevenZxy/CIS400/tree/f69489c0624157ae86b5d8ddb1fa99c89a927256/code/LSTM-Neural-Network-for-Time-Series-Prediction-master
        #     """
        #     import pandas as pd
        #     dataframe = pd.read_csv(f"{dataset_dir}/sinewave.csv")
        #     test_size,seq_len = 1500, 50
        #     data_test = dataframe.get("sinewave").values[-(test_size + 50):]
        #     data_windows = []
        #     for i in range(test_size):
        #         data_windows.append(data_test[i:i + seq_len])
        #     data_windows = np.array(data_windows).astype(float).reshape((test_size,seq_len,1))
        #     data_windows = np.array(data_windows).astype(float)
        #     x_test = data_windows[:, :-1]
        #     y_test = data_windows[:, -1, [0]]
        # elif 'price' in exp:
            # """see more details in https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject"""
            # x_test, y_test = DataUtils.get_price_data(dataset_dir)

        # TODO: Add your own data preprocessing here
        # Note: The returned inputs should be preprocessed and labels should decoded as one-hot vector which could be directly feed in model.
        # Both of them should be returned in batch, e.g. shape like (1500,28,28,1) and (1500,10)
        # elif 'xxx' in exp:
        #     x_test, y_test = get_your_data(dataset_dir)

        return dataset, dataset_name

    @staticmethod
    def save_img_from_array(path,array,index,exp):
        im = Image.fromarray(array)
        #path = path.rstrip("/")
        #save_path = "{}/{}_{}.png".format(path,exp,index)
        save_path = os.path.join(path,"{}_{}.png".format(exp, index))
        im.save(save_path)
        return save_path

    @staticmethod
    def shuffled_data(x, y, bs=None):
        ds = x.shape[0]
        all_idx = np.arange(ds)
        np.random.shuffle(all_idx)
        shuffle_idx = all_idx
        # shuffle_idx = all_idx[:bs]
        return x[shuffle_idx], y[shuffle_idx]

    @staticmethod
    def get_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test

    @staticmethod
    def get_fashion_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 28, 28
        x_test = x_test.reshape(x_test.shape[0], w, h, 1)
        return x_test

    @staticmethod
    #in ms: NCDHW
    def get_cifar10_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 32, 32
        x_test = x_test.reshape(x_test.shape[0], w, h, 3)
        return x_test
    
    @staticmethod
    # in ms: NCHW, NCDHW
    def get_cifar100_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        h, w = 32, 32
        x_test = x_test.reshape(x_test.shape[0], 3, h, w)

    @staticmethod
    def get_price_data(data_dir):
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        input_file = os.path.join(data_dir,"DIS.csv")
        df = pd.read_csv(input_file, header=None, index_col=None, delimiter=',')
        all_y = df[5].values
        dataset = all_y.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.5)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1, timestep 240
        look_back = 240
        trainX, trainY = create_dataset(train, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        return trainX,trainY

    @staticmethod
    def imagenet_preprocess_dict():
        import keras
        keras_preprocess_dict = dict()
        keras_preprocess_dict['resnet50'] = keras.applications.resnet50.preprocess_input
        keras_preprocess_dict['densenet121'] = keras.applications.densenet.preprocess_input
        keras_preprocess_dict['mobilenet.1.00.224'] = keras.applications.mobilenet.preprocess_input
        keras_preprocess_dict['vgg16'] = keras.applications.vgg16.preprocess_input
        keras_preprocess_dict['vgg19'] = keras.applications.vgg19.preprocess_input
        keras_preprocess_dict['inception.v3'] = keras.applications.inception_v3.preprocess_input
        keras_preprocess_dict['inception.v2'] = keras.applications.inception_resnet_v2.preprocess_input
        keras_preprocess_dict['xception'] = keras.applications.xception.preprocess_input
        return keras_preprocess_dict

    @staticmethod
    def imagenet_shape_dict():
        image_shapes = dict()
        image_shapes['resnet50'] = (224,224)
        image_shapes['densenet121'] = (224,224)
        image_shapes['mobilenet.1.00.224'] = (224,224)
        image_shapes['vgg16'] = (224,224)
        image_shapes['vgg19'] = (224, 224)
        image_shapes['inception.v3'] = (299,299)
        image_shapes['inception.v2'] = (299, 299)
        image_shapes['xception'] = (299,299)
        return image_shapes


class ToolUtils:

    @staticmethod
    def select_mutant(roulette,**kwargs):
        return roulette.choose_mutant()

    @staticmethod
    def select_mutator(logic, **kwargs):
        # import numpy as np
        # return np.random.permutation(mutate_ops)[0]
        last_used_mutator = kwargs['last_used_mutator']
        return logic.choose_mutator(last_used_mutator)

    @staticmethod
    def get_HH_mm_ss(td):
        days, seconds = td.days, td.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs

    
    @staticmethod
    # symbolTree是模型的symboltree;
    # node是symbolTree中的节点
    # result表示此传入的symbolTree中一共有多少个节点，传入默认是0
    # mapping_index_node是字典，#key是数字索引，value是node，传入默认空字典
    # mapping_node_parent是字典，key是数字索引，value是数字索引对应node的parent_tree，传入时默认空字典
    def judge_node(symbolTree, result, mapping_index_node, mapping_node_parent):
        import mindspore
        for node in symbolTree.nodes():
            sub_tree = mindspore.rewrite.TreeNodeHelper.get_sub_tree(node)
            if sub_tree is None:
                result += 1
                parent_tree = symbolTree
                mapping_index_node[result - 1] = node
                mapping_node_parent[result - 1] = parent_tree
            else:
                parent_tree = sub_tree
                ToolUtils.judge_node(parent_tree, result, mapping_index_node, mapping_node_parent)
        return result, mapping_index_node, mapping_node_parent


    @staticmethod
    def judge_module(cell):
        cell_num = len(list(cell.cells_and_names()))
        if cell_num > 1:
            return True
        else:
            return False

    @staticmethod
    def get_layers(cell):
        result = 0
        mapping = dict()
        model_num = 0
        layers = cell.cells_and_names()

        # print(len(list(layers)))
        result = len(list(layers))
        layers = cell.cells_and_names()
        for i, layer in enumerate(layers):
            if ToolUtils.judge_module(layer[1]):
                result -= 1
            else:
                mapping[model_num] = i
                model_num += 1
        assert result == model_num
        return result, mapping

class MetricsUtils:
    
    @staticmethod
    def concat_dataset(dataset, dataset_name, test_size):
        import mindspore
        label_tensor = None
        label_list = []
        if dataset_name == "cifar100":
            main_logger.info("Concat the label_Tensor of dataset cifar100!")
            for i, data in enumerate(dataset.create_dict_iterator()):
                label_tensor = data['fine_label']
                label_tensor = mindspore.numpy.expand_dims(label_tensor, 0)
                label_list.append(label_tensor)
                break
            for i, data in enumerate(dataset.create_dict_iterator()):
                if i == 0: 
                    continue
                data = data['fine_label']
                data = mindspore.numpy.expand_dims(data, 0)
                label_list.append(label_tensor)
                # print(np.shape(data))
                if i == test_size-1:
                    break
            label_tensor = mindspore.ops.concat(label_list)
            return label_tensor
        elif dataset_name == 'cifar10':
            main_logger.info("Concat the label_Tensor of dataset cifar10!")
            for i, data in enumerate(dataset.create_dict_iterator()):
                label_tensor = data['label']
                label_tensor = mindspore.numpy.expand_dims(label_tensor, 0)
                break
            for i, data in enumerate(dataset.create_dict_iterator()):
                if i == 0:
                    continue
                data = data['label']
                data = mindspore.numpy.expand_dims(data, 0)
                # print(np.shape(data))
                label_tensor = mindspore.ops.concat((label_tensor, data))
                if i == test_size-1:
                    break
            #print(label_tensor.shape)
            return label_tensor

    @staticmethod
    # def delta(y1_pred, y2_pred,y_true=None):
    #     y1_pred = np.reshape(y1_pred, [np.shape(y1_pred)[0], -1])
    #     y2_pred = np.reshape(y2_pred, [np.shape(y2_pred)[0], -1])
    #     return np.mean(np.abs(y1_pred - y2_pred), axis=1), np.sum(np.abs(y1_pred - y2_pred), axis=1)
    def delta(y1_pred, label_tensor, y_true=None):
        import mindspore
        # label_tensor = MetricsUtils.concat_dataset(dataset, dataset_name, test_size)
        y1_pred = np.reshape(y1_pred, [np.shape(y1_pred)[0], -1])
        #np.reshape will change the datatype to object.
        # label_tensor = mindspore.Tensor(label_tensor, dtype=mindspore.float32)
        label_tensor = label_tensor.asnumpy()
        # label_tensor = np.reshape(label_tensor, [np.shape(label_tensor)[0], -1])
        # now the datatypes of y1_pred and label_tensor are all object
        # print(np.shape(y1_pred))
        # print(np.shape(label_tensor))
        mean_ans = np.mean(np.abs(y1_pred - label_tensor), axis = 1)
        sum_ans = np.sum(np.abs(y1_pred - label_tensor), axis=1)
        # main_logger.info("the value of mean_ans: {}".format(mean_ans))
        # main_logger.info("the value of sum_ans: {}".format(sum_ans))
        return mean_ans, sum_ans

    # @staticmethod
    # # metrics_func(prediction1, prediction2, y_test[:flags.test_size])
    # def D_MAD_metrics(y1_pred, y2_pred,y_true, epsilon=1e-7):
    #     # sum could be remove and use mean in branch.
    #     theta_y1,sum_y1 = MetricsUtils.delta(y1_pred, y_true)
    #     theta_y2,sum_y2 = MetricsUtils.delta(y2_pred, y_true)
    #     return [
    #         0
    #         if (sum_y1[i] == 0 and sum_y2[i] == 0)
    #         else
    #         np.abs(theta_y1[i] - theta_y2[i]) / (theta_y1[i] + theta_y2[i])
    #         for i in range(len(y_true))
    #     ]
    @staticmethod
    # prediction1, prediction2, dataset, dataset_name, test_size
    def D_MAD_metrics(y1_pred, y2_pred,dataset, dataset_name, test_size, epsilon=1e-7):
        # sum could be remove and use mean in branch.
        label_tensor = MetricsUtils.concat_dataset(dataset, dataset_name, test_size)
        theta_y1,sum_y1 = MetricsUtils.delta(y1_pred, label_tensor)
        theta_y2,sum_y2 = MetricsUtils.delta(y2_pred, label_tensor)
        # if all(sum_y1 == 0) and all(sum_y2 == 0):
        #     return 0
        # else:
        #     return np.abs(theta_y1 - theta_y2) / (theta_y1 + theta_y2)
        var = [
            0
            if (sum_y1[i] == 0 and sum_y2[i] == 0)
            else
            np.abs(theta_y1[i] - theta_y2[i]) / (theta_y1[i] + theta_y2[i])
            for i in range(len(label_tensor))
        ]
        # main_logger.info("D_MAD result: {}".format(var))
        return var
            

    @staticmethod
    def get_all_metrics():
        metrics_dict = {}
        metrics_dict['D_MAD'] = MetricsUtils.D_MAD_metrics
        return metrics_dict

    @staticmethod
    def get_metrics_by_name(name):
        metrics = MetricsUtils.get_all_metrics()
        return metrics[name]

    @staticmethod
    def generate_result_by_metrics(metrics_list,lemon_results,save_dir,exp):

        for metrics_name in metrics_list:
            file_name = "{}/{}_{}_result.csv".format(save_dir,exp,metrics_name)
            metrics_result_dict = lemon_results[metrics_name]
            with open(file_name, "w") as writer:
                writer.write("Mutation-Backend-Pair,Inconsistency Score\n")
                for dm_k,dm_v in metrics_result_dict.items():
                    writer.write("{},{}\n".format(dm_k,dm_v))

# the class below is defined for the intermediate files interlayer mutation operator
class Node(object):
    def __init__(self, index, unique_name, shape,
                 operator_name=None, node_module=None, input_list=None,
                 output_list=None, output_name=None, copy_num=None,
                 ast_index=None, ms_class=None):
        self.index = index
        self.unique_name = unique_name  # module name + output name
        self.operator_name = operator_name
        self.node_module = node_module
        self.input_list = input_list
        self.output_list = output_list
        self.shape = shape
        self.output_name = output_name
        self.copy_num = copy_num
        self.ast_index = ast_index
        self.ms_class = ms_class  # newly added by hyr

    def set_uniquename(self, unique_name):
        self.unique_name = unique_name

    def set_operator_name(self, operator_name):
        self.operator_name = operator_name

    def set_input(self, input_list):
        self.input_list = input_list

    def set_output(self, output_list):
        self.output_list = output_list

    def set_module(self, module):
        self.node_module = module

    def set_copy_num(self, copy_num):
        self.copy_num = copy_num

    def set_ast_index(self, ast_index):
        self.ast_index = ast_index

    def set_ms_operator(self, ms_class):  # newly added by hyr
        self.ms_class = ms_class

    def get_prefix(self):
        unique_name \
            = self.unique_name
        prefix = unique_name.split(".")
        prefix = '.'.join(prefix[:-1])
        return prefix

BinOpTable = {
    "<class '_ast.Add'>": "add",
    "class '_ast.Sub'": "sub",
    "class '_ast.Mult'": "mult",
    "class '_ast.Div'": "div",
    "class '_ast.FloorDiv'": "floordiv",
    "class '_ast.Mod'": "mod",
    "class '_ast.Pow'": "pow",
    "class '_ast.LShift'": "lshift",
    "class '_ast.RShift'": "rshift",
    "class '_ast.BitOr'": "bitor",
    "class '_ast.BitXor'": "bitxor",
    "class '_ast.BitAnd'": "bitand",
    "class '_ast.MatMult'": "matmult"
}

class Table(object):
    def __init__(self, model_ast):
        self.nodeList = dict()
        self.ast = model_ast
        self.add_node_num = 0

    def add_node(self, node):
        self.nodeList[node.index] = node

    def print(self):
        for item in self.nodeList:
            item = self.nodeList[item]
            print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module,
                  '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list,
                  "+++++", item.copy_num, "+++++", item.ast_index)

    def print_nodelist(self, indices):
        keys = self.nodeList.keys()
        if isinstance(indices, list):
            for index in indices:
                item = self.nodeList[index]
                print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module,
                      '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++',
                      item.output_list,
                      "+++++", item.copy_num, "+++++", item.ast_index)
        elif isinstance(indices, int):
            item = self.nodeList[indices]
            print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module,
                  '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list,
                  "+++++", item.copy_num, "+++++", item.ast_index)

    def node_list_len(self):
        return len(self.nodeList)

    def save_ast(self, save_path):
        code_str = astunparse.unparse(self.ast)
        with open(save_path, 'w') as f:
            f.write(code_str)

    @staticmethod
    def construct_module_dict(model_ast):
        module_dict = dict()
        for item in model_ast.body:
            if isinstance(item, ast.ClassDef):
                module_dict[item.name] = item
        return module_dict

    @staticmethod
    def construct_table(model_ast, analyzed_data, module_dict):
        class MyNodeVisitor(ast.NodeVisitor):
            def __init__(self):
                super(MyNodeVisitor, self).__init__()

            def visit_Assign(self, node: ast.Assign):
                pass

        table = Table(model_ast)
        for i, data_item in enumerate(analyzed_data):
            node = Node(index=i, unique_name=data_item[1], shape=data_item[2], output_name=data_item[0])

            module_name = Table.find_module(module_dict, node.unique_name)
            copy_num = Table.get_copy_name(module_name)
            unique_name, operator_name = Table.get_name(data_item)
            node.set_uniquename(unique_name)
            node.set_operator_name(operator_name)
            node.set_module(module_name)
            node.set_copy_num(copy_num)
            node.set_input(data_item[3])
            return_list = Table.get_model_index(data_item[3], analyzed_data)
            node.set_input(return_list)

            table.add_node(node)
        output_list = Table.get_model_output_index(analyzed_data)
        for index in table.nodeList.keys():
            ast_index = get_ast_index(table, index)
            table.nodeList[index].set_ast_index(ast_index)
            table.nodeList[index].set_output(output_list[index])
        return table

    @staticmethod
    def get_name(data_item):
        unique_name, module_name = data_item[0], data_item[1]
        module_names = module_name.split(".")
        module_name = module_names[-1]
        if len(module_names) > 1:
            unique_name = '.'.join(module_names[:-1]) + '.' + unique_name
        return unique_name, module_name

    @staticmethod
    def get_copy_name(module_list):
        if not module_list:
            return None
        copy_name = list()
        for i in range(len(module_list)):
            copy_name.append(0)
        return copy_name

    @staticmethod
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
                        module = assign.value.func.id
                        module_list = Table.deep_find_module(module_dict, module, unique_names[1:], module_list)
                        return module_list
            return None

    @staticmethod
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
                                module_list = Table.deep_find_module(module_dict, module, unique_names[1:], module_list)
                                return module_list
        print("ERROR: Not find corresbonding module {}".format(unique_names))
        return None

    @staticmethod
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
                input_name = input_tuple[-1]  # 最后一位是输入的name
                input_prefix = input.rstrip("." + input_name)  # 前缀用来筛选，防止出现相同的name
                # print(input_name)
                # print(input_prefix)
                # print("===========")
                for i, element in enumerate(analyzed_data):
                    if (input_name != 'x') and (input_name == element[0]) and (input_prefix in element[1]):
                        return_list.append(i)
                        break
                    elif (input_name == 'x') and (input_prefix in element[1]):  # 往上找到第一个的input的index;
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
            if 'ast' in data_element[1]:  # 处理一些特殊情况；
                input_search = output_name
            elif len(op_prefix) != 0:
                input_search = op_prefix + "." + output_name
            else:
                input_search = output_name
            # 默认analyzed_data的最后一条元素是最终的输出；
            if data_element == analyzed_data[-1]:
                return_list.append(-2)  # 表示最终输出张量；
            # 先考虑相同class之内的；
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

        # 再考虑class跳转出去的，以及其他特殊情况.初步打算倒着遍历；
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


def search_init_ms_class(table, index):  # newly added by hyr
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


def search_init_statement(table, index):  # search_init_statement
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
                    if isinstance(init_func.body[j], ast.Assign):
                        if isinstance(init_func.body[j].targets[0], ast.Attribute) and init_func.body[j].targets[0].attr == prefix[i]:
                            return_list.append(j)
                        elif isinstance(init_func.body[j].targets[0], ast.Name):
                            print(init_func.body[j].targets[0].id)
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

def insert_node(table, index, param_dict, new_node_name=None):
    '''
    insert a new node after the node ordered in index
    if new_node_name is None, we copy the index node
    :param table:
    :param index:
    :param new_node_name:
    :param param_dict:
    :return:
    '''
    target_node = table.nodeList[index]
    # if the module num is more than mindsporeModel, we need copy the module
    if len(target_node.node_module) > 1:
        table, param_dict = copy_module(table=table, index=index, param_dict=param_dict)
    #get output shape from index
    assert len(target_node.output_list) == 1, "The length of output in current op should be one"
    output_shape = target_node.shape
    print(output_shape)
    kwargs = {"input_shape": output_shape}
    # insert the node after the index
    layer_utils = LayerUtils()
    op_name = "self.addNode_" + str(table.add_node_num)
    out_name = "opt_addNode_" + str(table.add_node_num)
    if new_node_name in layer_utils.available_model_level_layers.keys():
        insert_str = layer_utils.available_model_level_layers[new_node_name](**kwargs)
        insert_str = insert_str[0]
    else:
        print("{} not implemented!".format(new_node_name))
        return None
        # get op_name
    table.add_node_num += 1
    # get full inser str
    if not new_node_name is None:
        insert_str = op_name + " = " + insert_str
        init_insert_node = ast.parse(insert_str).body[0]
    # get out_opt_name of target node
    out_target_name = target_node.unique_name.split(".")[-1]
    insert_str2 = out_name + " = self." + op_name + "( {} )".format(out_target_name)
    construct_insert_node = ast.parse(insert_str2).body[0]
    # insert the node
    # in ast, direct add node after the statement
    for module in table.ast.body:
        if isinstance(module, ast.ClassDef) and module.name == target_node.node_module[-1]:
            assert len(target_node.input_list) == 1
            # 结点在init函数中的位置用
            ast_index = target_node.ast_index[-1]
            init_func = module.body[0]
            if not new_node_name is None:
                module.body[0].body.insert(ast_index[0], init_insert_node)
            else:
                # copy target_node
                init_insert_node = init_func.body[ast_index[0]].value
                init_str = astunparse.unparse(init_insert_node) # 这句话有问题
                init_str = op_name + " = " + init_str
                init_insert_node = ast.parse(init_insert_node).body[0]
                module.body[0].body.insert(ast_index[0], init_insert_node)
            # 默认插入的module位置和上一行相同，如果正好插入在module里面的倒数第二行，则还需要修改最后一行
            # 在construct函数中找到target node的位置，
            con_func = module.body[1]
            con_func.body.insert(ast_index[1], construct_insert_node)
            if ast_index[1] == len(con_func.body) - 3:
                # need modify the return statement
                module.body[1].body[-1].value.id = out_name

    node_len = table.node_list_len()
    insert_index = node_len

    # update table info
    # need update unique_name, output_name
    target_unique_name = target_node.unique_name
    prefix = ".".join(target_unique_name.split(".")[:-1])
    unique_name = prefix + "." + out_name
    # print(unique_name)

    add_node = Node(index=insert_index, unique_name=unique_name, shape=target_node.shape, operator_name = op_name, node_module=target_node.node_module, input_list=[target_node.index],
                 output_list=target_node.output_list, output_name=out_name, copy_num=target_node.copy_num,
                 ast_index = None, ms_class = None)
    print(add_node)
    table.add_node(add_node)
    # new_node = table.nodeList[insert_index]
    # change the info of the target node and the node after the add node
    origin_output_list = target_node.output_list
    table.nodeList[index].output_list = [table.nodeList[insert_index].index]
    for i in origin_output_list:
        table.nodeList[i].input_list = [table.nodeList[insert_index].index]

    for index in table.nodeList.keys():
        ast_index = get_ast_index(table, index)
        table.nodeList[index].set_ast_index(ast_index)
    # update param_dict
    param_new_list = get_new_param(irtable=table, param_dict=param_dict, layer_name=new_node_name, index=insert_index)
    # get_new_param(irtable, param_dict, layer_name, index, op_name, prefix)
    return table, param_new_list



def insert_mode_MLA(table, index, new_node_name, param_dict):
    target_node = table.nodeList[index]
    # if the module num is more than mindsporeModel, we need copy the module
    if len(target_node.node_module) > 1:
        table, model_ast = copy_module(table=table, index=index, param_dict=param_dict)

    # get output shape from index
    assert len(target_node.output_list) == 1, "The length of output in current op should be one"
    output_shape = target_node.output_list[0]
    kwargs = {"input_shape": output_shape}

    layer_utils = LayerUtils()
    if new_node_name in layer_utils.available_model_level_layers.keys():
        insert_str = layer_utils.available_model_level_layers[new_node_name](**kwargs)





def replace_node(table, index, new_node):
    raise NotImplementedError

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




def get_weight_shape(layer_name, input_shape):
    # （batch、in_channels、kernel_wide、kernel_height）
    input_shape_list = list(input_shape)
    # contain transpose
    if "conv_1d" in layer_name:
        return {"weight": [input_shape_list[1], input_shape_list[1], 3],
                "bias": [input_shape_list[1]]}
    # contain transpose
    elif "conv_2d" in layer_name:
        return {"weight": [input_shape_list[1], input_shape_list[1], 3, 3],
                "bias": [input_shape_list[1]]}
    # contain transpose
    elif "conv_3d" in layer_name:
        return {"weight": [input_shape_list[1], input_shape_list[1], 3, 3, 3],
                "bias": [input_shape_list[1]]}
    elif "dense" in layer_name:
        return {"weight": [input_shape_list[1], input_shape_list[1]],
                "bias": [input_shape_list[1]]}
    elif "depthwise_conv_2d" in layer_name:
        return {"weight": [input_shape_list[1], 1, input_shape_list[2], input_shape_list[3]],
                "bias": [input_shape_list[1]]}
    elif "separable_conv_1d" in layer_name:
        return [{"weight": [input_shape_list[1], 1, input_shape_list[2]],
                 "bias": [input_shape_list[1]]},
                {"weight": [input_shape_list[1], input_shape_list[1], input_shape_list[2]],
                 "bias": [input_shape_list[1]]}]
    elif "separable_conv_2d" in layer_name:
        return [{"weight": [input_shape_list[1], 1, input_shape_list[2], input_shape_list[3]],
                 "bias": [input_shape_list[1]]},
                {"weight": [input_shape_list[1], input_shape_list[1], input_shape_list[2], input_shape_list[3]],
                 "bias": [input_shape_list[1]]}]
    # contain 1d、2d、3d
    elif "batch_normalization" in layer_name:
        return {"gamma": [input_shape_list[1]], "beta": [input_shape_list[1]], "moving_mean": [input_shape_list[1]],
                "moving_variance": [input_shape_list[1]]}
    else:
        print("Do not support the layer {}".format(layer_name))
        return None


def get_param(param_dict, layer_name, input_shape):
    # ckpt_path = f'../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt'
    # layer_name = "conv2d"
    dict_shape = get_weight_shape(layer_name, input_shape) # (16, 16, 3, 3)
    # param_dict = mindspore.load_checkpoint(ckpt_path)
    param_weight_list = []
    for key in param_dict.keys():
        for l in list(Tensor(param_dict[key]).flatten().asnumpy()):
            param_weight_list.append(l)
    mean = np.mean(param_weight_list)
    std = np.std(param_weight_list)
    np.random.seed(0)
    weight_np = np.random.normal(mean, std, dict_shape["weight"])
    bias_np = np.random.normal(mean, std, dict_shape["bias"])
    param_weight = Parameter(Tensor(weight_np, mindspore.float32), name="test_weight", requires_grad=True)
    param_bias = Parameter(Tensor(bias_np, mindspore.float32), name="test_bias", requires_grad=True)
    return param_weight, param_bias


# 去做权重的部分：把天杰学长写的“获得新加层的权重。权重的生成算法在原来的lemon里面有”。这边需要对接下。
# get_weight_shape：可以知道当前插入层各种参数的形状；
# handle_weight: 往param_dict中加入新的kv
# 之后根据lemon获得数值，计算正态分布。
# 获取全部的weight——变成一个list——得到正态分布——再进行随机采样——采样的数就作为新插入的层的参数。
def get_new_param(irtable, param_dict, layer_name, index): # newly added by hyr
    nodeList = irtable.nodeList
    error_flag = 1
    param_new_list = []  # param_new_list会复制param_dict的所有已有kv，并加入新层的参数，之后返回
    for node_index in nodeList.keys():
        node = nodeList[node_index]
        if(node.index == index):
            error_flag = 0
            input_index_list = node.input_list
            input_node = nodeList[input_index_list[0]]
            input_shape = input_node.shape
            new_param_weight, new_param_bias= get_param(param_dict, layer_name, input_shape)
            #参数保存
            unique_name = node.unique_name
            weight_name = unique_name + '.weight'
            print(weight_name)
            bias_name = unique_name + '.bias'
            param_new_list.append({"name": "{}".format(weight_name), "data": new_param_weight})
            param_new_list.append({"name": "{}".format(bias_name), "data": new_param_bias})
            for key in param_dict.keys():
                # 复制param_dict的所有已有kv
                param_new = {}
                param_new["name"] = key
                param_new["data"] = param_dict[key]
                param_new_list.append(param_new)
                # 加入新层的参数
            break
        else:
            continue
    if(error_flag == 1):
        print("ERROR: layer {} is not in the table.")
    return param_new_list


if __name__ == '__main__':
    pass





