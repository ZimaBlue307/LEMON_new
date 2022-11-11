import os
import pickle
import math
from PIL import Image
import warnings
import datetime
import configparser
import numpy as np
from scripts.logger.lemon_logger import Logger

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
        from scripts.mutation.mutation_utils import LayerUtils
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
    #一些operators在三个测试的mindspore版本中，api不一样。故本函数的参数多传入一个backend信息以作判断
    def get_data_by_exp_with_bk(exp, test_size, backend_name, cfg_name): #like exp = alexnet_cifar100
        """
        old: return x_test and y_test
        new: return dataset after certain operations
        """
        import mindspore
        # import keras
        # import keras.backend as K
        # K.set_image_data_format("channels_last")
        # K.set_image_data_format("channels_first")
        # in mindspore, data format is channel_first, so we may need to skip up three lines.

        lemon_cfg = configparser.ConfigParser()
        # lemon_cfg.read("./config/experiments.conf")
        # lemon_cfg.read(cfg_name)
        # dataset_dir = lemon_cfg['parameters']['dataset_dir']
        x_test = y_test = []
        # adding new elif branch
        if 'cifar100' in exp:
            dataset_name = "cifar100"
            cifar100_dir = "dataset/cifar100/cifar-100-binary"
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
                
            else:
                one_hot_opt = transforms.OneHot(num_classes=100) 
                rescale_op = CV.Rescale(rescale_param, shift_param)
                resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
            #In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label"(100) and "coarse_label"(20)
                dataset = dataset.map(operations = one_hot_opt, input_columns=["fine_label"]) #把细标签转换为独热编码   
                dataset = dataset.map(operations = rescale_op, input_columns=["image"])
                dataset = dataset.map(operations = resize_op, input_columns=["image"])
            for i, data in enumerate(dataset.create_dict_iterator()):
                label_shape = data['fine_label'].shape
                break
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
    def judge_node(symbolTree, node, result, mapping_index_node, mapping_node_parent):
        import mindspore
        sub_tree = mindspore.rewrite.TreeNodeHelper.get_sub_tree(node)
        if sub_tree is None:
            result += 1
            parent_tree = symbolTree
            mapping_index_node[result - 1] = node
            mapping_node_parent[result - 1] = parent_tree
            return result
        else:
            parent_tree = sub_tree
            for sub_node in parent_tree.nodes():
                result = ToolUtils.judge_node(sub_tree, sub_node, result, mapping_index_node, mapping_node_parent)
            return result


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
        if dataset_name == "cifar100":
            for i, data in enumerate(dataset.create_dict_iterator()):
                label_tensor = data['fine_label']
                label_tensor = mindspore.numpy.expand_dims(label_tensor, 0)
                break
            for i, data in enumerate(dataset.create_dict_iterator()):
                if i == 0: 
                    continue
                data = data['fine_label']
                data = mindspore.numpy.expand_dims(data, 0)
                # print(np.shape(data))
                label_tensor = mindspore.ops.concat((label_tensor, data))
                if i == test_size-1:
                    break
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
        label_tensor = mindspore.Tensor(label_tensor, dtype=mindspore.float32)
        # label_tensor = np.reshape(label_tensor, [np.shape(label_tensor)[0], -1])
        # now the datatypes of y1_pred and label_tensor are all object
        # print(np.shape(y1_pred))
        # print(np.shape(label_tensor))
        mean_ans = np.mean(np.abs(y1_pred - label_tensor), axis = 1)
        sum_ans = np.sum(np.abs(y1_pred - label_tensor), axis=1)
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
        return [
            0
            if (sum_y1[i] == 0 and sum_y2[i] == 0)
            else
            np.abs(theta_y1[i] - theta_y2[i]) / (theta_y1[i] + theta_y2[i])
            for i in range(len(label_tensor))
        ]
            

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

if __name__ == '__main__':
    pass





