import numpy as np
import os
import warnings
np.random.seed(20200501)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ActivationUtils:
    def __init__(self):
        self.available_activations = ActivationUtils.available_activations()

    @staticmethod
    def available_activations():
        activations = {}
        import mindspore.nn as nn
        activations['relu'] = nn.ReLU
        activations['tanh'] = nn.Tanh
        activations['sigmoid'] = nn.Sigmoid
        activations['no_activation'] = ActivationUtils.no_activation
        activations['leakyrelu'] = ActivationUtils.leakyrelu
        return activations

    def get_activation(self, activation):
        if activation not in self.available_activations.keys():
            raise Exception('Activation function {} is not supported. Supported functions: {}'
                            .format(activation, [key for key in self.available_activations.keys()]))
        return self.available_activations[activation]

    def pick_activation_randomly(self, activations=None):
        if activations is None:
            availables = [item for item in self.available_activations.keys()]
            availables.remove('no_activation')
        else:
            availables = activations
        index = np.random.randint(0, len(availables))
        return self.available_activations[availables[index]]

    @staticmethod
    def no_activation(x):
        return x

    @staticmethod
    def leakyrelu(x):
        import mindspore.nn as nn
        leaky_relu = nn.LeakyReLU(alpha = 0.01)
        output = leaky_relu(x)
        return output 

class LayerUtils:
    def __init__(self):
        # these layers take effect both for training and testing
        self.available_model_level_layers = {}
        # these layers only take effect for training
        self.available_source_level_layers = {}
        self.is_input_legal = {}

        self.available_model_level_layers['dense'] = LayerUtils.dense
        self.is_input_legal['dense'] = LayerUtils.dense_input_legal
        self.available_model_level_layers['conv_1d'] = LayerUtils.conv1d
        self.is_input_legal['conv_1d'] = LayerUtils.conv1d_input_legal
        self.available_model_level_layers['conv_2d'] = LayerUtils.conv2d
        self.is_input_legal['conv_2d'] = LayerUtils.conv2d_input_legal
        self.available_model_level_layers['separable_conv_1d'] = LayerUtils.separable_conv_1d
        self.is_input_legal['separable_conv_1d'] = LayerUtils.separable_conv_1d_input_legal
        self.available_model_level_layers['separable_conv_2d'] = LayerUtils.separable_conv_2d
        self.is_input_legal['separable_conv_2d'] = LayerUtils.separable_conv_2d_input_legal
        self.available_model_level_layers['depthwise_conv_2d'] = LayerUtils.depthwise_conv_2d
        self.is_input_legal['depthwise_conv_2d'] = LayerUtils.depthwise_conv_2d_input_legal
        self.available_model_level_layers['conv_2d_transpose'] = LayerUtils.conv_2d_transpose
        self.is_input_legal['conv_2d_transpose'] = LayerUtils.conv_2d_transpose_input_legal
        self.available_model_level_layers['conv_3d'] = LayerUtils.conv_3d
        self.is_input_legal['conv_3d'] = LayerUtils.conv_3d_input_legal
        self.available_model_level_layers['conv_3d_transpose'] = LayerUtils.conv_3d_transpose
        self.is_input_legal['conv_3d_transpose'] = LayerUtils.conv_3d_transpose_input_legal
        self.available_model_level_layers['max_pooling_1d'] = LayerUtils.max_pooling_1d
        self.is_input_legal['max_pooling_1d'] = LayerUtils.max_pooling_1d_input_legal
        self.available_model_level_layers['max_pooling_2d'] = LayerUtils.max_pooling_2d
        self.is_input_legal['max_pooling_2d'] = LayerUtils.max_pooling_2d_input_legal
        self.available_model_level_layers['max_pooling_3d'] = LayerUtils.max_pooling_3d
        self.is_input_legal['max_pooling_3d'] = LayerUtils.max_pooling_3d_input_legal
        self.available_model_level_layers['average_pooling_1d'] = LayerUtils.average_pooling_1d
        self.is_input_legal['average_pooling_1d'] = LayerUtils.average_pooling_1d_input_legal
        self.available_model_level_layers['average_pooling_2d'] = LayerUtils.average_pooling_2d
        self.is_input_legal['average_pooling_2d'] = LayerUtils.average_pooling_2d_input_legal
        self.available_model_level_layers['average_pooling_3d'] = LayerUtils.average_pooling_3d
        self.is_input_legal['average_pooling_3d'] = LayerUtils.average_pooling_3d_input_legal
        self.available_model_level_layers['batch_normalization'] = LayerUtils.batch_normalization
        self.is_input_legal['batch_normalization'] = LayerUtils.batch_normalization_input_legal
        self.available_model_level_layers['leaky_relu_layer'] = LayerUtils.leaky_relu_layer
        self.is_input_legal['leaky_relu_layer'] = LayerUtils.leaky_relu_layer_input_legal
        self.available_model_level_layers['prelu_layer'] = LayerUtils.prelu_layer
        self.is_input_legal['prelu_layer'] = LayerUtils.prelu_layer_input_legal
        self.available_model_level_layers['elu_layer'] = LayerUtils.elu_layer
        self.is_input_legal['elu_layer'] = LayerUtils.elu_layer_input_legal
        self.available_model_level_layers['thresholded_relu_layer'] = LayerUtils.thresholded_relu_layer
        self.is_input_legal['thresholded_relu_layer'] = LayerUtils.thresholded_relu_layer_input_legal
        self.available_model_level_layers['softmax_layer'] = LayerUtils.softmax_layer
        self.is_input_legal['softmax_layer'] = LayerUtils.softmax_layer_input_legal
        self.available_model_level_layers['relu_layer'] = LayerUtils.relu_layer
        self.is_input_legal['relu_layer'] = LayerUtils.relu_layer_input_legal

        self.available_source_level_layers['activity_regularization_l1'] = LayerUtils.activity_regularization_l1
        self.is_input_legal['activity_regularization_l1'] = LayerUtils.activity_regularization_input_legal
        self.available_source_level_layers['activity_regularization_l2'] = LayerUtils.activity_regularization_l1
        self.is_input_legal['activity_regularization_l2'] = LayerUtils.activity_regularization_input_legal

    def is_layer_in_weight_change_white_list(self, layer):
        #import keras
        import mindspore
        white_list = [mindspore.nn.Dense, mindspore.nn.Conv1d, mindspore.nn.Conv2d, mindspore.nn.Conv3d,
                      #keras.layers.DepthwiseConv2D,
                      mindspore.nn.Conv2dTranspose, mindspore.nn.Conv3dTranspose,
                      mindspore.nn.MaxPool1D, mindspore.nn.MaxPool, mindspore.ops.MaxPool3D,
                      mindspore.nn.AvgPool1d, mindspore.nn.AvgPool2d, mindspore.ops.AvgPool3D,
                      mindspore.nn.LeakyReLU, mindspore.nn.ELU, keras.layers.ThresholdedReLU,
                      mindspore.nn.Softmax, mindspore.ops.ReLU
                      ]
        #in mindspore1.7.0, DepthwiseConv2dNative don't have any description files since it will be deprecated in the future, 
        #and the developer recommand us to use Conv2D instead.
        #keras.layers.AveragePooling3D——>mindspore.nn.AvgPool3D
        #keras.layers.AveragePooling2D——>mindspore.nn.AvgPool
        # print(white_list)
        for l in white_list:
            if isinstance(layer, l):
                return True
        return False

    @staticmethod
    def clone(layer):
        from scripts.tools.utils import ModelUtils
        custom_objects = ModelUtils.custom_objects()
        layer_config = layer.get_config()
        if 'activation' in layer_config.keys():
            activation = layer_config['activation']
            if activation in custom_objects.keys():
                layer_config['activation'] = 'relu'
                clone_layer = layer.__class__.from_config(layer_config)
                clone_layer.activation = custom_objects[activation]
            else:
                clone_layer = layer.__class__.from_config(layer_config)
        else:
            clone_layer = layer.__class__.from_config(layer_config)
        return clone_layer
        # return layer.__class__.from_config(layer.get_config())

    @staticmethod
    def dense(input_shape):
        # input_shape = input_shape.as_list()
        import mindspore
        layer = mindspore.nn.Dense(input_shape[-1], input_shape=(input_shape[1:],))
        # may need to change the input
        layer.name += '_insert'
        return layer

    @staticmethod
    def dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None

    @staticmethod
    def conv1d(input_shape):
        import mindspore
        #layer = keras.layers.Conv1D(input_shape[-1], 3, strides=1, padding='same')
        #When using this layer as the first layer in a model, 
        #provide an input_shape argument (tuple of integers or None;
        # 3 is kernel size
        layer = mindspore.nn.Conv1d(input_shape[-1], kernel_size=3, stride=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def conv2d(input_shape):
        # input_shape = input_shape.as_list()
        import mindspore
        #layer = keras.layers.Conv2D(input_shape[-1], 3, strides=(1,1), padding='same')
        layer = mindspore.nn.Conv2d(input_shape[-1], kernel_size=3, strides=(1,1), pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def separable_conv_1d(input_shape):
        import keras
        layer = keras.layers.SeparableConv1D(input_shape[-1], 3, strides=1, padding='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def separable_conv_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def separable_conv_2d(input_shape):
        import keras
        layer = keras.layers.SeparableConv2D(input_shape[-1], 3, strides=(1,1), padding='same')
        #深度可分离的二维卷积。
        layer.name += '_insert'
        return layer

    @staticmethod
    def separable_conv_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def depthwise_conv_2d(input_shape):
        import keras
        layer = keras.layers.DepthwiseConv2D(3, strides=(1,1), padding='same')
        #暂时不知道如何修改这个深度卷积的API，先放着
        layer.name += '_insert'
        return layer

    @staticmethod
    def depthwise_conv_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_2d_transpose(input_shape):
        import mindspore
        layer = mindspore.nn.Conv2dTranspose(input_shape[-1], kernel_size = 3, stride=(1,1), pad_mode='same')
        #not for sure what does input_shape[-1] means, means filters?
        #回头再改
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv_2d_transpose_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_3d(input_shape):
        import mindspore
        #layer = keras.layers.Conv3D(input_shape[-1], 3, strides=(1,1,1), padding='same')
        layer = mindspore.nn.Conv3d(input_shape[-1], kernel_size = 3, stride=(1,1,1), pad_mode='same')
        #what's the in_channel and out_channel?
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def conv_3d_transpose(input_shape):
        import mindspore
        #layer = keras.layers.Conv3DTranspose(input_shape[-1], 3, strides=(1,1,1), padding='same')
        layer = mindspore.nn.Conv3dTranspose(input_shape[-1], kernel_size = 3, stride=(1,1,1), pad_mode='same')
        #not for sure what does input_shape[-1] means
        #回头再改
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv_3d_transpose_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def max_pooling_1d(input_shape):
        import mindspore
        layer = mindspore.nn.MaxPool1d(kernel_size=3, stride=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def max_pooling_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def max_pooling_2d(input_shape):
        import mindspore
        layer = mindspore.ops.MaxPool2d(kernel_size=(3, 3), stride=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def max_pooling_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def max_pooling_3d(input_shape):
        import mindspore
        layer = mindspore.ops.MaxPool3D(kernel_size=(3, 3, 3), strides=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def max_pooling_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def average_pooling_1d(input_shape):
        import mindspore
        layer = mindspore.nn.AvgPool1d(kernel_size=3, stride=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def average_pooling_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def average_pooling_2d(input_shape):
        import mindspore
        layer = mindspore.nn.AvgPool2d(kernel_size=(3, 3), stride=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def average_pooling_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def average_pooling_3d(input_shape):
        import mindspore
        layer = mindspore.ops.AvgPool3D(kernel_size=(3, 3, 3), strides=1, pad_mode='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def average_pooling_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def batch_normalization(input_shape):
        import mindspore
        #layer = keras.layers.BatchNormalization(input_shape=input_shape[1:])
        #keras里默认is training 是true
        layer = mindspore.ops.BatchNorm(is_training  = True, )
        layer.name += '_insert'
        return layer

    @staticmethod
    def batch_normalization_input_legal(input_shape):
        return True

    @staticmethod
    def leaky_relu_layer(input_shape):
        import mindspore.nn as nn
        leaky_relu = nn.LeakyReLU(alpha = 0.3)
        layer = leaky_relu(input_shape=input_shape[1:])
        layer.name += '_insert'
        return layer

    @staticmethod
    def leaky_relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def prelu_layer(input_shape):
        import mindspore
        #layer = keras.layers.PReLU(input_shape=input_shape[1:], alpha_initializer='RandomNormal')
        layer = mindspore.ops.PReLU(input_shape=input_shape[1:], weight='RandomNormal')
        #不确定改的对不对
        layer.name += '_insert'
        return layer

    @staticmethod
    def prelu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def elu_layer(input_shape):
        import mindspore
        elu = mindspore.nn.ELU()
        layer = elu(input_shape=input_shape[1:])
        layer.name += '_insert'
        return layer

    @staticmethod
    def elu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def thresholded_relu_layer(input_shape):
        #ThresholdedReLU: f(x) = x for x > theta otherwise f(x) = 0
        #修改办法应该和def relu_layer(input_shape)相同，回头来修改
        import keras
        layer = keras.layers.ThresholdedReLU(input_shape=input_shape[1:])
        layer.name += '_insert'
        return layer

    @staticmethod
    def thresholded_relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def softmax_layer(input_shape):
        import mindspore
        layer = mindspore.nn.Softmax(input_shape=input_shape[1:])
        layer.name += '_insert'
        return layer

    @staticmethod
    def softmax_layer_input_legal(input_shape):
        return True

    @staticmethod
    def relu_layer(input_shape):
        import mindspore
        relu = mindspore.ops.ReLU()
        layer = relu(input_shape=input_shape[1:])
        #没有修改relu中的max-value=1.0，回头来修改
        layer.name += '_insert'
        return layer

    @staticmethod
    def relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def activity_regularization_l1(input_shape):
        import keras
        layer = keras.layers.ActivityRegularization(l1=0.5, l2=0.0)
        layer.name += '_insert'
        return layer

    @staticmethod
    def activity_regularization_l2(input_shape):
        import keras
        layer = keras.layers.ActivityRegularization(l1=0.0, l2=0.5)
        layer.name += '_insert'
        return layer

    @staticmethod
    def activity_regularization_input_legal(input_shape):
        return True


if __name__ == '__main__':
    # activation_utils = ActivationUtils()
    # result = activation_utils.pick_activation_randomly(['relu', 'leakyrelu'])
    # print(result)
    layerUtils = LayerUtils()
    result = layerUtils.is_layer_in_weight_change_white_list(layerUtils.available_model_level_layers['dense']([None, 3]))
    print(result)