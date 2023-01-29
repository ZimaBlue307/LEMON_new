#assuming all the input_shapes are channel first;
#layer_matching in lemon_new
import os
import warnings
import mindspore
from mindspore import nn
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ResizeBilinear(nn.Cell):
    def __init__(self, out_shape):
        super(ResizeBilinear, self).__init__()
        self.out_shape = out_shape
        self.ResizeBilinear = mindspore.nn.ResizeBilinear()

    def construct(self, x):
        result = self.ResizeBilinear(x, self.out_shape)
        return result

class LayerMatching:
    concat_size_limit = 1e4

    def __init__(self):
        self.layers = {}
        self.constraints = {}

        # self.layers['flatten'] = LayerMatching.flatten
        # self.constraints['flatten'] = LayerMatching.flatten_constraints

        self.layer_concats = {}
        self.input_legal = {}
        self.layer_concats['Flatten'] = LayerMatching.Flatten
        self.input_legal['Flatten'] = LayerMatching.Flatten_input_legal
        # self.layer_concats['cropping1d'] = LayerMatching.cropping1d_dense
        # self.input_legal['cropping1d'] = LayerMatching.cropping1d_dense_input_legal
        # self.layer_concats['cropping2d'] = LayerMatching.cropping2d_dense
        # self.input_legal['cropping2d'] = LayerMatching.cropping2d_dense_input_legal
        # self.layer_concats['cropping3d'] = LayerMatching.cropping3d_dense
        # self.input_legal['cropping3d'] = LayerMatching.cropping3d_dense_input_legal
        # self.layer_concats['upsampling_1d'] = LayerMatching.upsampling_1d_dense
        # self.input_legal['upsampling_1d'] = LayerMatching.upsampling_1d_dense_input_legal
        # self.layer_concats['upsampling_2d'] = LayerMatching.upsampling_2d_dense
        # self.input_legal['upsampling_2d'] = LayerMatching.upsampling_2d_dense_input_legal
        # self.layer_concats['upsampling_3d'] = LayerMatching.upsampling_3d_dense
        # self.input_legal['upsampling_3d'] = LayerMatching.upsampling_3d_dense_input_legal
        self.layer_concats['ConstantPad1d'] = LayerMatching.ConstantPad1d
        self.input_legal['ConstantPad1d'] = LayerMatching.ConstantPad1d_input_legal
        self.layer_concats['ConstantPad2d'] = LayerMatching.ConstantPad2d
        self.input_legal['ConstantPad2d'] = LayerMatching.ConstantPad2d_input_legal
        self.layer_concats['ConstantPad3d'] = LayerMatching.ConstantPad3d
        self.input_legal['ConstantPad3d'] = LayerMatching.ConstantPad3d_input_legal
        self.layer_concats['AdaptiveAvgPool1d'] = LayerMatching.AdaptiveAvgPool1d
        self.input_legal['AdaptiveAvgPool1d'] = LayerMatching.AdaptiveAvgPool1d_input_legal
        self.layer_concats['AdaptiveMaxPool1d'] = LayerMatching.AdaptiveMaxPool1d
        self.input_legal['AdaptiveMaxPool1d'] = LayerMatching.AdaptiveMaxPool1d_input_legal
        self.layer_concats['AdaptiveMaxPool2d'] = LayerMatching.AdaptiveMaxPool2d
        self.input_legal['AdaptiveMaxPool2d'] = LayerMatching.AdaptiveMaxPool2d_input_legal
        self.layer_concats['Pad'] = LayerMatching.Pad
        self.input_legal['Pad'] = LayerMatching.Pad_input_legal
        self.layer_concats['ReflectionPad1d'] = LayerMatching.ReflectionPad1d
        self.input_legal['ReflectionPad1d'] = LayerMatching.ReflectionPad1d_input_legal
        self.layer_concats['ReflectionPad2d'] = LayerMatching.ReflectionPad2d
        self.input_legal['ReflectionPad2d'] = LayerMatching.ReflectionPad2d_input_legal
        # self.layer_concats['ReflectionPad3d'] = LayerMatching.ReflectionPad3d
        # self.input_legal['ReflectionPad3d'] = LayerMatching.ReflectionPad3d_input_legal
        self.layer_concats['ZeroPad2d'] = LayerMatching.ZeroPad2d
        self.input_legal['ZeroPad2d'] = LayerMatching.ZeroPad2d_input_legal
        # self.layer_concats['GroupNorm'] = LayerMatching.GroupNorm
        # self.input_legal['GroupNorm'] = LayerMatching.GroupNorm_input_legal
        self.layer_concats['LayerNorm'] = LayerMatching.LayerNorm
        self.input_legal['LayerNorm'] = LayerMatching.LayerNorm_input_legal
        # self.layer_concats['global_max_pooling_1d'] = LayerMatching.global_max_pooling_1d_dense
        # self.input_legal['global_max_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        # self.layer_concats['global_average_pooling_1d'] = LayerMatching.global_average_pooling_1d_dense
        # self.input_legal['global_average_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        # self.layer_concats['global_max_pooling_2d'] = LayerMatching.global_max_pooling_2d_dense
        # self.input_legal['global_max_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        # self.layer_concats['global_average_pooling_2d'] = LayerMatching.global_average_pooling_2d_dense
        # self.input_legal['global_average_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        # self.layer_concats['global_max_pooling_3d'] = LayerMatching.global_max_pooling_3d_dense
        # self.input_legal['global_max_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        # self.layer_concats['global_average_pooling_3d'] = LayerMatching.global_average_pooling_3d_dense
        # self.input_legal['global_average_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        # self.layer_concats['simple_rnn'] = LayerMatching.simple_rnn_dense
        # self.input_legal['simple_rnn'] = LayerMatching.simple_rnn_dense_input_legal
        # self.layer_concats['gru'] = LayerMatching.gru_dense
        # self.input_legal['gru'] = LayerMatching.gru_dense_input_legal
        # self.layer_concats['lstm'] = LayerMatching.lstm_dense
        # self.input_legal['lstm'] = LayerMatching.lstm_dense_input_legal
        # self.layer_concats['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense
        # self.input_legal['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense_input_legal

    #done
    # @staticmethod
    # def flatten(input_shape):
    #     #import keras
    #     #return keras.layers.Flatten()
    #     return mindspore.nn.Flatten()
        
    # #done
    # @staticmethod
    # def flatten_constraints(input_shape):
    #     #input_shape = input_shape.as_list()
    #     input_shape = list(input_shape)
    #     input_shape_len = len(input_shape)
    #     constraints = []
    #     if input_shape_len < 2:
    #         return None
    #     constraints = []
    #     dim_size = 1
    #     for i in range(input_shape_len):
    #         if i == 0:
    #             continue
    #         constraints.append('= input_{} {}'.format(i, input_shape[i]))
    #         dim_size *= input_shape[i]
    #     constraint_str = '= output_{} {}'.format(1, dim_size)
    #     constraints.append(constraint_str)
    #     return constraints


    #不知道怎么实现reshape层
    # @staticmethod
    # def flatten_dense(input_shape):
    #
    #     # units = 1
    #     # for i in range(len(input_shape)):
    #     #     if i == 0:
    #     #         continue #jump batch_size
    #     #     units *= input_shape[i]
    #
    #     layer_str1 = "nn.Flatten()"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #
    #     # layer_str2 = "nn.Dense(in_channels={}, out_channels={})".format(units, units)
    #
    #     # layer_str3 = "layer_matching_mindspore.Reshape({})".format(input_shape[1:])
    #
    #     return [layer_str1, layer_str2]

    @staticmethod
    def Flatten(input_shape):
        module_str = "class Flatten_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(Flatten_ResizeBilinear_Module, self).__init__() \
                            self.flatten = nn.Flatten() \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_flatten = self.flatten(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_flatten, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    #done
    @staticmethod
    def Flatten_input_legal(input_shape):
        input_shape = list(input_shape)
        is_legal = len(input_shape) > 3 and input_shape[1] is not None and input_shape[2] is not None
        # concat_size = 1
        # for i, dim in enumerate(input_shape):
        #     if i == 0:
        #         continue
        #     is_legal = is_legal and dim is not None
        #     if dim is not None:
        #         concat_size *= dim
        return is_legal

    # @staticmethod
    # def cropping1d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #未修改
    #     layer_concat.append(keras.layers.Cropping1D(cropping=(1, 1)))
    #     # layer_concat.append(mindspore.dataset.vision.c_transforms.Crop((0,1), ()))#don't know what to do yet
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1]))
    #     return layer_concat
    #
    # @staticmethod
    # def cropping1d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] > 2 \
    #            and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def cropping2d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #layer_concat.append(keras.layers.Cropping2D(cropping=((1, 1), (1, 1))))
    #     #cropping = tuple of 2 tuples of 2 ints:
    #     #interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
    #     layer_concat.append(mindspore.dataset.vision.c_transforms.Crop(coordinates = (1, 1), size = 1)) #not for sure
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, ((input_shape[1] - 2) * (input_shape[2] - 2) * input_shape[3])))
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def cropping2d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 4 and input_shape[0] is None \
    #            and input_shape[1] is not None and input_shape[1] > 2 \
    #            and input_shape[2] is not None and input_shape[2] > 2 \
    #            and input_shape[3] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def cropping3d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #未修改
    #     layer_concat.append(keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1))))#don't know what to do yet
    #     #mindspore.dataset.vision.c_transforms.Crop, but not for sure yet.
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, ((input_shape[1] - 2) * (input_shape[2] - 2) * (input_shape[3] - 2) * input_shape[4])))
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def cropping3d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 5 and input_shape[0] is None \
    #            and input_shape[1] is not None and input_shape[1] > 2 \
    #            and input_shape[2] is not None and input_shape[2] > 2 \
    #            and input_shape[3] is not None and input_shape[3] > 2 \
    #            and input_shape[4] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    # @staticmethod
    # def upsampling_1d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.UpSampling1D(size=2))
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     return layer_concat
    #
    # @staticmethod
    # def upsampling_1d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
    #            and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def upsampling_2d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.UpSampling2D(size=(2, 2)))
    #     layer_concat.append(mindspore.nn.ResizeBilinear(input_shape[1:]))
    #     # layer_concat.append(mindspore.nn.Flatten())
    #     # layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3]))
    #     # layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def upsampling_2d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 4 and input_shape[0] is None \
    #            and input_shape[1] is not None and input_shape[2] is not None and input_shape[3] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def upsampling_3d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.UpSampling3D(size=(2, 2, 2)))
    #     #mindspore.nn.ResizeBilinear
    #     #仅支持bilinear模式对数据进行采样
    #     layer_concat.append(mindspore.nn.Flatten())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def upsampling_3d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 5 and input_shape[0] is None \
    #            and input_shape[1] is not None \
    #            and input_shape[2] is not None \
    #            and input_shape[3] is not None \
    #            and input_shape[4] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

#8.8
    # @staticmethod
    # def ConstantPad1d(input_shape):
    #     layer_str1 = "nn.ConstantPad1d(padding=1, value=0)"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def ConstantPad1d(input_shape):
        module_str = "class ConstantPad1d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ConstantPad1d_ResizeBilinear_Module, self).__init__() \
                            self.constantPad1d = nn.ConstantPad1d(padding=1, value=0) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_constantPad1d = self.constantPad1d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_constantPad1d, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    @staticmethod
    def ConstantPad1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None

    @staticmethod
    def ConstantPad2d(input_shape):
        module_str = "class ConstantPad2d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ConstantPad2d_ResizeBilinear_Module, self).__init__() \
                            self.constantPad2d = nn.ConstantPad2d(padding=1, value=0) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_constantPad2d = self.constantPad2d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_constantPad2d, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    # @staticmethod
    # def ConstantPad2d(input_shape):
    #     layer_str1 = "nn.ConstantPad2d(padding=1, value=0)"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def ConstantPad2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None

    @staticmethod
    def ConstantPad3d(input_shape):
        module_str = "class ConstantPad3d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ConstantPad3d_ResizeBilinear_Module, self).__init__() \
                            self.constantPad3d = nn.ConstantPad3d(padding=1, value=0) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_constantPad3d = self.constantPad3d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_constantPad3d, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    # @staticmethod
    # def ConstantPad3d(input_shape):
    #     layer_str1 = "nn.ConstantPad3d(padding=1, value=0)"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def ConstantPad3d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None

    @staticmethod
    def AdaptiveAvgPool1d(input_shape):
        module_str = "class AdaptiveAvgPool1d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(AdaptiveAvgPool1d_ResizeBilinear_Module, self).__init__() \
                            self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(output_size={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_adaptiveAvgPool1d = self.adaptiveAvgPool1d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_adaptiveAvgPool1d, {}) \
                            return opt_resizeBilinear".format(input_shape[2:], input_shape)
        return module_str

    # @staticmethod
    # def AdaptiveAvgPool1d(input_shape):
    #     layer_str1 = "nn.AdaptiveAvgPool1d(output_size={})".format(input_shape[2:])
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def AdaptiveAvgPool1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None

    # @staticmethod
    # def AdaptiveMaxPool1d(input_shape):
    #     layer_str1 = "nn.AdaptiveMaxPool1d(output_size={})".format(input_shape[2:])
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def AdaptiveMaxPool1d(input_shape):
        module_str = "class AdaptiveMaxPool1d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(AdaptiveMaxPool1d_ResizeBilinear_Module, self).__init__() \
                            self.adaptiveMaxPool1d = nn.AdaptiveMaxPool1d(output_size={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_adaptiveMaxPool1d = self.adaptiveMaxPool1d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_adaptiveMaxPool1d, {}) \
                            return opt_resizeBilinear".format(input_shape[2:], input_shape)
        return module_str

    @staticmethod
    def AdaptiveMaxPool1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None

    @staticmethod
    def AdaptiveMaxPool2d(input_shape):
        module_str = "class AdaptiveMaxPool2d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(AdaptiveMaxPool2d_ResizeBilinear_Module, self).__init__() \
                            self.adaptiveMaxPool2d = nn.AdaptiveMaxPool2d(output_size={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_adaptiveMaxPool2d = self.adaptiveMaxPool2d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_adaptiveMaxPool2d, {}) \
                            return opt_resizeBilinear".format(input_shape[2:], input_shape)
        return module_str

    # @staticmethod
    # def AdaptiveMaxPool2d(input_shape):
    #     layer_str1 = "nn.AdaptiveMaxPool2d(output_size={})".format(input_shape[2:])
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def AdaptiveMaxPool2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None \
            and input_shape[2] is not None \
            and input_shape[3] is not None

    @staticmethod
    def Pad(input_shape):
        module_str = "class Pad_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(Pad_ResizeBilinear_Module, self).__init__() \
                            self.pad = nn.Pad(paddings={}, mode={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_pad = self.pad(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_pad, {}) \
                            return opt_resizeBilinear".format(1, "CONSTANT", input_shape)
        return module_str

    # # @staticmethod
    # # def AdaptiveMaxPool3d(input_shape):
    # #     layer_str1 = "nn.AdaptiveMaxPool3d(output_size={})".format(input_shape[2:])
    # #
    # #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    # #     return [layer_str1, layer_str2]

    @staticmethod
    def Pad_input_legal(input_shape):
        input_shape = list(input_shape)
        return True

    # @staticmethod
    # def ReflectionPad1d(input_shape):
    #     layer_str1 = "nn.ReflectionPad1d(padding=1)"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def ReflectionPad1d(input_shape):
        module_str = "class ReflectionPad1d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ReflectionPad1d_ResizeBilinear_Module, self).__init__() \
                            self.reflectionPad1d = nn.ReflectionPad1d(padding=1) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_reflectionPad1d = self.reflectionPad1d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_reflectionPad1d, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    @staticmethod
    def ReflectionPad1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None

    @staticmethod
    def ReflectionPad2d(input_shape):
        module_str = "class ReflectionPad2d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ReflectionPad2d_ResizeBilinear_Module, self).__init__() \
                            self.reflectionPad2d = nn.ReflectionPad2d(padding=1) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_reflectionPad2d = self.reflectionPad2d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_reflectionPad2d, {}) \
                            return opt_resizeBilinear".format(input_shape)
        return module_str

    # @staticmethod
    # def ReflectionPad2d(input_shape):
    #     layer_str1 = "nn.ReflectionPad2d(padding=1)"
    #
    #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    #     return [layer_str1, layer_str2]

    @staticmethod
    def ReflectionPad2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None

    @staticmethod
    def ZeroPad2d(input_shape):
        module_str = "class ZeroPad2d_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(ZeroPad2d_ResizeBilinear_Module, self).__init__() \
                            self.zeroPad2d = nn.ZeroPad2d(padding={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_zeroPad2d = self.zeroPad2d(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_zeroPad2d, {}) \
                            return opt_resizeBilinear".format(1, input_shape)
        return module_str


    @staticmethod
    def ZeroPad2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None

    @staticmethod
    def LayerNorm(input_shape):
        module_str = "class LayerNorm_ResizeBilinear_Module(nn.Cell): \
                        def __init__(self): \
                            super(LayerNorm_ResizeBilinear_Module, self).__init__() \
                            self.layerNorm = nn.LayerNorm(normalized_shape={}) \
                            self.resizeBilinear = mindspore.nn.ResizeBilinear() \
                        def construct(self, x): \
                            opt_layerNorm = self.layerNorm(x) \
                            opt_resizeBilinear = self.resizeBilinear(opt_layerNorm, {}) \
                            return opt_resizeBilinear".format(input_shape[1:], input_shape)
        return module_str


    @staticmethod
    def LayerNorm_input_legal(input_shape):
        input_shape = list(input_shape)
        return True

    # @staticmethod
    # def ReflectionPad3d(input_shape):
    #     module_str = "class ReflectionPad3d_ResizeBilinear_Module(nn.Cell): \
    #                     def __init__(self): \
    #                         super(ReflectionPad3d_ResizeBilinear_Module, self).__init__() \
    #                         self.reflectionPad3d = nn.ReflectionPad3d() \
    #                         self.resizeBilinear = mindspore.nn.ResizeBilinear() \
    #                     def construct(self, x): \
    #                         opt_reflectionPad3d = self.reflectionPad3d(x) \
    #                         opt_resizeBilinear = self.resizeBilinear(opt_reflectionPad3d, {}) \
    #                         return opt_resizeBilinear".format(input_shape)
    #     return module_str
    #
    # # @staticmethod
    # # def ReflectionPad3d(input_shape):
    # #     layer_str1 = "nn.ReflectionPad3d(padding=1)"
    # #
    # #     layer_str2 = "layer_matching_mindspore.ResizeBilinear({})".format(input_shape)
    # #     return [layer_str1, layer_str2]
    #
    # @staticmethod
    # def ReflectionPad3d_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 5 and input_shape[0] is None \
    #            and input_shape[1] is not None \
    #            and input_shape[2] is not None \
    #            and input_shape[3] is not None \
    #            and input_shape[4] is not None
    # @staticmethod
    # def global_max_pooling_1d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalMaxPooling1D())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_average_pooling_1d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalAveragePooling1D())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_pooling_1d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
    #            and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def global_max_pooling_2d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalMaxPooling2D())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_average_pooling_2d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalAveragePooling2D())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_pooling_2d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 4 and input_shape[0] is None \
    #            and input_shape[1] is not None \
    #            and input_shape[2] is not None \
    #            and input_shape[3] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def global_max_pooling_3d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalMaxPooling3D())
    #     layer_concat.append(mindspore.nn.Flatten())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_average_pooling_3d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     layer_concat.append(keras.layers.GlobalAveragePooling3D())
    #     layer_concat.append(mindspore.nn.Flatten())
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def global_pooling_3d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 5 and input_shape[0] is None \
    #            and input_shape[1] is not None \
    #            and input_shape[2] is not None \
    #            and input_shape[3] is not None \
    #            and input_shape[4] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    # @staticmethod
    # def simple_rnn_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #未修改
    #     layer_concat.append(keras.layers.SimpleRNN(50))
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def simple_rnn_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None \
    #            and input_shape[1] is not None \
    #            and input_shape[2] is not None \
    #            and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def gru_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #layer_concat.append(keras.layers.GRU(50))
    #     layer_concat.append(mindspore.nn.GRU(50))
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def gru_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
    #            and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def lstm_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #layer_concat.append(keras.layers.LSTM(50))
    #     layer_concat.append(mindspore.ops.LSTM(50)) #what does 50 means ?
    #     layer_concat.append(mindspore.nn.Dense(input_shape[-1], input_shape[1] * input_shape[2]))
    #     layer_concat.append(mindspore.ops.Reshape(input_shape, input_shape[1:]))
    #     return layer_concat
    #
    # @staticmethod
    # def lstm_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
    #            and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit
    #
    # @staticmethod
    # def conv_lstm_2d_dense(input_shape):
    #     import mindspore
    #     layer_concat = []
    #     #未修改
    #     layer_concat.append(keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', return_sequences=True))
    #     return layer_concat
    #
    # @staticmethod
    # def conv_lstm_2d_dense_input_legal(input_shape):
    #     input_shape = list(input_shape)
    #     return len(input_shape) == 5 and input_shape[0] is None and input_shape[1] is not None \
    #            and input_shape[2] is not None and input_shape[2] > 3 \
    #            and input_shape[3] is not None and input_shape[3] > 3 \
    #            and input_shape[4] is not None \
    #            and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit


if __name__ == '__main__':
    pass