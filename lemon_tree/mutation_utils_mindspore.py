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
        self.available_model_level_layers['conv_1d_transpose'] = LayerUtils.conv_1d_transpose
        self.is_input_legal['conv_1d_transpose'] = LayerUtils.conv_1d_transpose_input_legal
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
        self.available_model_level_layers['conv_3d'] = LayerUtils.conv3d
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
        self.available_model_level_layers['batch_normalization_1d'] = LayerUtils.batch_normalization_1d
        self.is_input_legal['batch_normalization_1d'] = LayerUtils.batch_normalization_1d_input_legal
        self.available_model_level_layers['batch_normalization_2d'] = LayerUtils.batch_normalization_2d
        self.is_input_legal['batch_normalization_2d'] = LayerUtils.batch_normalization_2d_input_legal
        self.available_model_level_layers['batch_normalization_3d'] = LayerUtils.batch_normalization_3d
        self.is_input_legal['batch_normalization_3d'] = LayerUtils.batch_normalization_3d_input_legal
        self.available_model_level_layers['leaky_relu_layer'] = LayerUtils.leaky_relu_layer
        self.is_input_legal['leaky_relu_layer'] = LayerUtils.leaky_relu_layer_input_legal
        self.available_model_level_layers['prelu_layer'] = LayerUtils.prelu_layer
        self.is_input_legal['prelu_layer'] = LayerUtils.prelu_layer_input_legal
        self.available_model_level_layers['elu_layer'] = LayerUtils.elu_layer
        self.is_input_legal['elu_layer'] = LayerUtils.elu_layer_input_legal
        self.available_model_level_layers['softmax_layer'] = LayerUtils.softmax_layer
        self.is_input_legal['softmax_layer'] = LayerUtils.softmax_layer_input_legal
        self.available_model_level_layers['relu_layer'] = LayerUtils.relu_layer
        self.is_input_legal['relu_layer'] = LayerUtils.relu_layer_input_legal
        # 扩展
        # self.available_model_level_layers['celu_layer'] = LayerUtils.celu_layer
        # self.available_model_level_layers['gelu_layer'] = LayerUtils.gelu_layer
        # self.available_model_level_layers['glu_layer'] = LayerUtils.glu_layer
        # self.available_model_level_layers['fastgelu_layer'] = LayerUtils.fastgelu_layer
        # self.available_model_level_layers['Hardtanh_layer'] = LayerUtils.Hardtanh_layer
        # self.available_model_level_layers['HShrink_layer'] = LayerUtils.HShrink_layer
        # self.available_model_level_layers['HSigmoid_layer'] = LayerUtils.HSigmoid_layer
        # self.available_model_level_layers['HSwish_layer'] = LayerUtils.HSwish_layer
        # self.available_model_level_layers['LogSigmoid_layer'] = LayerUtils.LogSigmoid_layer
        # self.available_model_level_layers['LogSoftmax_layer'] = LayerUtils.LogSoftmax_layer

        # self.available_model_level_layers['LRN_layer'] = LayerUtils.LRN_layer
        # self.available_model_level_layers['Mish_layer'] = LayerUtils.Mish_layer
        # self.available_model_level_layers['Softsign_layer'] = LayerUtils.Softsign_layer
        # self.available_model_level_layers['rrelu_layer'] = LayerUtils.rrelu_layer
        # self.available_model_level_layers['selu_layer'] = LayerUtils.selu_layer
        # self.available_model_level_layers['silu_layer'] = LayerUtils.silu_layer
        # self.available_model_level_layers['Sigmoid_layer'] = LayerUtils.Sigmoid_layer
        # self.available_model_level_layers['Softmin_layer'] = LayerUtils.Softmin_layer
        # self.available_model_level_layers['SoftShrink_layer'] = LayerUtils.SoftShrink_layer
        # self.available_model_level_layers['Tanh_layer'] = LayerUtils.Tanh_layer
        # self.available_model_level_layers['Tanhshrink_layer'] = LayerUtils.Tanhshrink_layer
        # self.available_model_level_layers['Threshold_layer'] = LayerUtils.Threshold_layer


    @staticmethod
    def dense(input_shape,
              weight_init='normal', bias_init='zeros', has_bias='True', activation='None'):
        layer_str = "nn.Dense(in_channels={}, out_channels={}, " \
                    "weight_init='{}', bias_init='{}', has_bias={}, " \
                    "activation={})".format(input_shape[1], input_shape[1], weight_init,
                                            bias_init, has_bias, activation)

        return [layer_str]

    @staticmethod
    def dense_input_legal(input_shape):
        #input_shape = input_shape.as_list()
        #as_list(): in keras, Convert a tuple to a list, report an error when it's not a tuple
        input_shape = list(input_shape)
        #print(input_shape)
        return len(input_shape) == 2 and input_shape[0] is not None and input_shape[1] is not None

    @staticmethod
    def conv1d(input_shape,
               kernel_size=3, stride=1, pad_mode="same",
               padding=0, dilation=1, group=1, has_bias='False',
               weight_init="normal", bias_init="zeros"):
        layer_str = "nn.Conv1d(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                           stride, pad_mode, padding, dilation, group,
                                           has_bias, weight_init, bias_init)

        return [layer_str]

    @staticmethod
    def conv1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_1d_transpose(input_shape,
                          kernel_size=3, stride=1, pad_mode="same",
                          padding=0, dilation=1, group=1, has_bias='False',
                          weight_init="normal", bias_init="zeros"):
        layer_str = "nn.Conv1dTranspose(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                           stride, pad_mode, padding, dilation, group,
                                           has_bias, weight_init, bias_init)

        return [layer_str]

    @staticmethod
    def conv_1d_transpose_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv2d(input_shape,
               kernel_size=3, stride=1, pad_mode="same",
               padding=0, dilation=1, group=1, has_bias=False,
               weight_init="normal", bias_init="zeros", data_format="NCHW"):
        layer_str = "nn.Conv2d(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                           stride, pad_mode, padding, dilation, group,
                                                           has_bias, weight_init, bias_init, data_format)

        return [layer_str]

    @staticmethod
    def conv2d_input_legal(input_shape):
        #input_shape = input_shape.as_list()
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def separable_conv_1d(input_shape,
                          kernel_size=3, group=1, stride=1, pad_mode="same",
                          padding=0, dilation=1, has_bias=False,
                          weight_init="normal", bias_init="zeros"):
        # in_channels、out_channels、group三个参数相等
        layer_str1 = "nn.Conv1d(in_channels={}, out_channels={}, " \
                     "kernel_size={}, stride={}, pad_mode='{}', " \
                     "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                     "bias_init='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                            stride, pad_mode, padding, dilation, input_shape[1],
                                            has_bias, weight_init, bias_init)

        # kernel_size = 1，group回到默认值1
        layer_str2 = "nn.Conv1d(in_channels={}, out_channels={}, " \
                     "kernel_size={}, stride={}, pad_mode='{}', " \
                     "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                     "bias_init='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                            stride, pad_mode, padding, dilation, group,
                                            has_bias, weight_init, bias_init)

        return [layer_str1, layer_str2]

    @staticmethod
    def separable_conv_1d_demo(input_shape,
                          kernel_size=3, group=1, stride=1, pad_mode="same",
                          padding=0, dilation=1, has_bias=False,
                          weight_init="normal", bias_init="zeros"):
        module_str = "class Seperable_Conv_Module(nn.Cell): \
                        def __init__(self): \
                            super(Seperable_Conv_Module, self).__init__() \
                            self.conv2d_0 = nn.Conv1d(in_channels={}, out_channels={}, " \
                                                "kernel_size={}, stride={}, pad_mode='{}', " \
                                                "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                                                "bias_init='{}') \
                            self.conv2d_1 = nn.Conv1d(in_channels={}, out_channels={}, " \
                                                "kernel_size={}, stride={}, pad_mode='{}', " \
                                                "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                                                "bias_init='{}') \
                        def construct(self, x): \
                            opt_conv2d_0 = self.conv2d_0(x) \
                            opt_conv2d_1 = self.conv2d_1(opt_conv2d_0) \
                            return opt_conv2d_1".format(input_shape[1], input_shape[1], kernel_size,
                                            stride, pad_mode, padding, dilation, input_shape[1],
                                            has_bias, weight_init, bias_init, input_shape[1], input_shape[1], kernel_size,
                                            stride, pad_mode, padding, dilation, group,
                                            has_bias, weight_init, bias_init)
        return module_str

    @staticmethod
    def separable_conv_1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def separable_conv_2d(input_shape,
                          kernel_size=3, group=1, stride=1, pad_mode="same",
                          padding=0, dilation=1, has_bias=False,
                          weight_init="normal", bias_init="zeros", data_format="NCHW"):
        # in_channels、out_channels、group三个参数相等
        layer_str1 = "nn.Conv2d(in_channels={}, out_channels={}, " \
                     "kernel_size={}, stride={}, pad_mode='{}', " \
                     "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                     "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                            stride, pad_mode, padding, dilation, input_shape[1],
                                                            has_bias, weight_init, bias_init, data_format)

        # kernel_size = 1，group回到默认值1
        layer_str2 = "nn.Conv2d(in_channels={}, out_channels={}, " \
                     "kernel_size={}, stride={}, pad_mode='{}', " \
                     "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                     "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                            stride, pad_mode, padding, dilation, group,
                                                            has_bias, weight_init, bias_init, data_format)

        return [layer_str1, layer_str2]

    @staticmethod
    def separable_conv_2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
            and input_shape[3] is not None and input_shape[3] >= 3

    # in_channels、out_channels、group三个参数相等
    @staticmethod
    def depthwise_conv_2d(input_shape,
                          kernel_size=3, group=1, stride=1, pad_mode="same",
                          padding=0, dilation=1, has_bias=False,
                          weight_init="normal", bias_init="zeros", data_format="NCHW"):
        layer_str = "nn.Conv2d(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                           stride, pad_mode, padding, dilation, input_shape[1],
                                                           has_bias, weight_init, bias_init, data_format)

        return [layer_str]

    @staticmethod
    def depthwise_conv_2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def conv_2d_transpose(input_shape,
                          kernel_size=3, stride=1, pad_mode="same",
                          padding=0, dilation=1, group=1, has_bias=False,
                          weight_init="normal", bias_init="zeros"):
        layer_str = "nn.Conv2dTranspose(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                           stride, pad_mode, padding, dilation, group,
                                           has_bias, weight_init, bias_init)

        return [layer_str]

    @staticmethod
    def conv_2d_transpose_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def conv3d(input_shape,
               kernel_size=3, stride=1, pad_mode="same",
               padding=0, dilation=1, group=1, has_bias=False,
               weight_init="normal", bias_init="zeros", data_format="NCHW"):
        layer_str = "nn.Conv3d(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                           stride, pad_mode, padding, dilation, group,
                                                           has_bias, weight_init, bias_init, data_format)

        return [layer_str]

    @staticmethod
    def conv_3d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3 \
               and input_shape[4] is not None and input_shape[4] >= 3

    @staticmethod
    def conv_3d_transpose(input_shape,
                          kernel_size=3, stride=1, pad_mode="same",
                          padding=0, dilation=1, group=1, output_padding=0, has_bias=False,
                          weight_init="normal", bias_init="zeros", data_format="NCHW"):
        layer_str = "nn.Conv3dTranspose(in_channels={}, out_channels={}, " \
                    "kernel_size={}, stride={}, pad_mode='{}', " \
                    "padding={}, dilation={}, group={}, output_padding={}, has_bias={}, weight_init='{}', " \
                    "bias_init='{}', data_format='{}')".format(input_shape[1], input_shape[1], kernel_size,
                                                           stride, pad_mode, padding, dilation, group, output_padding,
                                                           has_bias, weight_init, bias_init, data_format)

        return [layer_str]

    @staticmethod
    def conv_3d_transpose_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3 \
               and input_shape[4] is not None and input_shape[4] >= 3

    @staticmethod
    def max_pooling_1d(kernel_size=1, stride=1, pad_mode="same"):
        layer_str = "nn.MaxPool1d(kernel_size={}, stride={}, " \
                    "pad_mode='{}')".format(kernel_size, stride, pad_mode)

        return [layer_str]

    @staticmethod
    def max_pooling_1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def max_pooling_2d(kernel_size=1, stride=1, pad_mode="same", data_format="NCHW"):
        layer_str = "nn.MaxPool2d(kernel_size={}, stride={}, " \
                    "pad_mode='{}', data_format='{}')".format(kernel_size, stride, pad_mode, data_format)

        return [layer_str]

    @staticmethod
    def max_pooling_2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def max_pooling_3d(kernel_size=3, stride=None, padding=0, dilation=1,
                       return_indices=False, ceil_mode=False):
        layer_str = "nn.MaxPool3d(kernel_size={}, stride={}, padding={}, " \
                    "dilation={}, return_indices={}, ceil_mode={})".format(kernel_size, stride, padding,
                                                                           dilation, return_indices, ceil_mode)

        return [layer_str] #这里好像有点问题呀；

    @staticmethod
    def max_pooling_3d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3 \
               and input_shape[4] is not None and input_shape[4] >= 3

    @staticmethod
    def average_pooling_1d(kernel_size=1, stride=1, pad_mode="same"):
        layer_str = "nn.AvgPool1d(kernel_size={}, stride={}, " \
                    "pad_mode='{}')".format(kernel_size, stride, pad_mode)

        return [layer_str]

    @staticmethod
    def average_pooling_1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def average_pooling_2d(kernel_size=1, stride=1, pad_mode="same", data_format="NCHW"):
        layer_str = "nn.AvgPool2d(kernel_size={}, stride={}, " \
                    "pad_mode='{}', data_format='{}')".format(kernel_size, stride, pad_mode, data_format)

        return [layer_str]

    @staticmethod
    def average_pooling_2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def average_pooling_3d(kernel_size=3, stride=None, padding=0, ceil_mode=False,
                           count_include_pad=True, divisor_override=None):
        layer_str = "nn.AvgPool3d(kernel_size={}, stride={}, ceil_mode={}, " \
                    "count_include_pad={}, divisor_override={})".format(kernel_size, stride, padding, ceil_mode,
                                                                        count_include_pad, divisor_override)

        return [layer_str]

    @staticmethod
    def average_pooling_3d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3 \
               and input_shape[4] is not None and input_shape[4] >= 3

    @staticmethod
    def batch_normalization_1d(input_shape, eps=1e-5, momentum=0.9, affine=True,
                               gamma_init='ones', beta_init='zeros', moving_mean_init='zeros',
                               moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'):
        layer_str = "nn.BatchNorm1d(num_features={}, eps={}, momentum={}, affine={}, " \
                    "gamma_init='{}', beta_init='{}', moving_mean_init='{}', moving_var_init='{}'," \
                    "use_batch_statistics={}, data_format='{}')".format(input_shape[1], eps, momentum, affine,
                                                                      gamma_init, beta_init, moving_mean_init,
                                                                      moving_var_init,
                                                                      use_batch_statistics, data_format)

        return [layer_str]

    @staticmethod
    def batch_normalization_1d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 3 and input_shape[1] is not None and input_shape[2] is not None

    @staticmethod
    def batch_normalization_2d(input_shape, eps=1e-5, momentum=0.9, affine=True,
                               gamma_init='ones', beta_init='zeros', moving_mean_init='zeros',
                               moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'):
        layer_str = "nn.BatchNorm2d(num_features={}, eps={}, momentum={}, affine={}, " \
                    "gamma_init='{}', beta_init='{}', moving_mean_init='{}', moving_var_init='{}'," \
                    "use_batch_statistics={}, data_format='{}')".format(input_shape[1], eps, momentum, affine,
                                                                      gamma_init, beta_init, moving_mean_init,
                                                                      moving_var_init,
                                                                      use_batch_statistics, data_format)

        return [layer_str]

    @staticmethod
    def batch_normalization_2d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None and input_shape[3] is not None

    @staticmethod
    def batch_normalization_3d(input_shape, eps=1e-5, momentum=0.9, affine=True,
                               gamma_init='ones', beta_init='zeros', moving_mean_init='zeros',
                               moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'):
        layer_str = "nn.BatchNorm3d(num_features={}, eps={}, momentum={}, affine={}, " \
                    "gamma_init='{}', beta_init='{}', moving_mean_init='{}',' moving_var_init='{}'," \
                    "use_batch_statistics={}, data_format='{}')".format(input_shape[1], eps, momentum, affine,
                                                                      gamma_init, beta_init, moving_mean_init,
                                                                      moving_var_init,
                                                                      use_batch_statistics, data_format)

        return [layer_str]

    @staticmethod
    def batch_normalization_3d_input_legal(input_shape):
        input_shape = list(input_shape)
        return len(input_shape) == 5 and input_shape[1] is not None and input_shape[2] is not None \
            and input_shape[3] is not None and input_shape[4] is not None


    @staticmethod
    def leaky_relu_layer(alpha=0.2):
        layer_str = "nn.LeakyReLU(alpha={})".format(alpha)

        return [layer_str]

    @staticmethod
    def leaky_relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def prelu_layer(channel=1, w=0.25):
        layer_str = "nn.PReLU(channel={}, w={})".format(channel, w)

        return [layer_str]

    @staticmethod
    def prelu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def elu_layer(alpha=1.0):
        layer_str = "nn.ELU(alpha={})".format(alpha)

        return [layer_str]

    @staticmethod
    def elu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def softmax_layer(axis=-1):
        layer_str = "nn.Softmax(axis={})".format(axis)

        return [layer_str]

    @staticmethod
    def softmax_layer_input_legal(input_shape):
        return True

    @staticmethod
    def relu_layer():
        layer_str = "nn.ReLU()".format()

        return [layer_str]

    @staticmethod
    def relu_layer_input_legal(input_shape):
        return True

    # @staticmethod
    # def AdaptiveMaxPool1d_layer(output_size):
    #     layer_str = "nn.AdaptiveMaxPool1d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def AdaptiveMaxPool2d_layer(output_size):
    #     layer_str = "nn.AdaptiveMaxPool2d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def AdaptiveMaxPool3d_layer(output_size):
    #     layer_str = "nn.AdaptiveMaxPool3d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def AdaptiveAvgPool1d_layer(output_size):
    #     layer_str = "nn.AdaptiveAvgPool1d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def AdaptiveAvgPool2d_layer(output_size):
    #     layer_str = "nn.AdaptiveAvgPool2d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def AdaptiveAvgPool3d_layer(output_size):
    #     layer_str = "nn.AdaptiveAvgPool3d(output_size={})".format(output_size)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def FractionalMaxPool2d_layer(kernel_size, output_size=None, output_ratio=None, return_indices=False,
    #                               _random_samples=None):
    #     layer_str = "nn.FractionalMaxPool2d(kernel_size={}, output_size={},output_ratio={}," \
    #                 "return_indices={},_random_samples={},)".format(kernel_size, output_size, output_ratio,
    #                                                                 return_indices, _random_samples)
    #
    #     return [layer_str]
    #
    # @staticmethod
    # def FractionalMaxPool3d_layer(kernel_size, output_size=None, output_ratio=None, return_indices=False,
    #                               _random_samples=None):
    #     layer_str = "nn.FractionalMaxPool3d(kernel_size={}, output_size={},output_ratio={}," \
    #                 "return_indices={},_random_samples={},)".format(kernel_size, output_size, output_ratio,
    #                                                                 return_indices, _random_samples)
    #
    #     return [layer_str]

    @staticmethod
    def celu_layer(alpha=1.0):
        layer_str = "nn.CELU(alpha={})".format(alpha)

        return [layer_str]

    @staticmethod
    def gelu_layer(approximate=True):
        layer_str = "nn.GELU(approximate={})".format(approximate)

        return [layer_str]

    @staticmethod
    def glu_layer(axis=-1):
        layer_str = "nn.GLU(axis={})".format(axis)

        return [layer_str]

    @staticmethod
    def fastgelu_layer():
        layer_str = "nn.FastGelu()".format()

        return [layer_str]

    @staticmethod
    def Hardtanh_layer(min_val=-1.0, max_val=1.0):
        layer_str = "nn.Hardtanh(min_val={}, max_val={})".format(min_val, max_val)

        return [layer_str]

    @staticmethod
    def HShrink_layer(lambd=0.5):
        layer_str = "nn.HShrink(lambd={})".format(lambd)

        return [layer_str]

    @staticmethod
    def HSigmoid_layer():
        layer_str = "nn.HSigmoid()".format()

        return [layer_str]

    @staticmethod
    def HSwish_layer():
        layer_str = "nn.HSwish()".format()

        return [layer_str]

    @staticmethod
    def LogSigmoid_layer():
        layer_str = "nn.LogSigmoid()".format()

        return [layer_str]

    @staticmethod
    def LogSoftmax_layer(axis=-1):
        layer_str = "nn.LogSoftmax(axis={})".format(axis)

        return [layer_str]
