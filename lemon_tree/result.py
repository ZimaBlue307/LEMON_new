
import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

class Module2(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_kernel_size, conv2d_0_padding, conv2d_0_pad_mode):
        super(Module2, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=16, kernel_size=conv2d_0_kernel_size, stride=(1, 1), padding=conv2d_0_padding, pad_mode=conv2d_0_pad_mode, dilation=(1, 1), group=1, has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1

class Module2_copy(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_kernel_size, conv2d_0_padding, conv2d_0_pad_mode):
        super(Module2_copy, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=16, kernel_size=conv2d_0_kernel_size, stride=(1, 1), padding=conv2d_0_padding, pad_mode=conv2d_0_pad_mode, dilation=(1, 1), group=1, has_bias=True)
        self.conv2d_0_copy = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=16, kernel_size=conv2d_0_kernel_size, stride=(1, 1), padding=conv2d_0_padding, pad_mode=conv2d_0_pad_mode, dilation=(1, 1), group=1, has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_conv2d_0_copy = self.conv2d_0_copy(opt_conv2d_0)
        opt_relu_1 = self.relu_1(opt_conv2d_0_copy)
        return opt_relu_1

class Module0(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features, conv2d_2_in_channels, conv2d_2_out_channels, conv2d_2_stride, conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels, out_channels=conv2d_2_out_channels, kernel_size=(1, 1), stride=conv2d_2_stride, padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels, out_channels=conv2d_4_out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', dilation=(1, 1), group=1, has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        return opt_relu_5

class Module5(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels, module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module5, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_0_conv2d_2_in_channels, conv2d_2_out_channels=module0_0_conv2d_2_out_channels, conv2d_2_stride=module0_0_conv2d_2_stride, conv2d_4_in_channels=module0_0_conv2d_4_in_channels, conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=conv2d_0_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_1_conv2d_2_in_channels, conv2d_2_out_channels=module0_1_conv2d_2_out_channels, conv2d_2_stride=module0_1_conv2d_2_stride, conv2d_4_in_channels=module0_1_conv2d_4_in_channels, conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels, out_channels=conv2d_2_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.add = mindspore.ops.Add()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = self.add(x, opt_conv2d_0)
        module0_1_opt = self.module0_1(opt_add_1)
        opt_conv2d_2 = self.conv2d_2(module0_1_opt)
        opt_add_3 = self.add(opt_add_1, opt_conv2d_2)
        return opt_add_3

class Module3(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels):
        super(Module3, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_0_conv2d_2_in_channels, conv2d_2_out_channels=module0_0_conv2d_2_out_channels, conv2d_2_stride=module0_0_conv2d_2_stride, conv2d_4_in_channels=module0_0_conv2d_4_in_channels, conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=conv2d_0_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        return opt_conv2d_0

class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module2_0 = Module2_1(conv2d_0_in_channels=3, conv2d_0_kernel_size=(3, 3), conv2d_0_padding=(1, 1, 1, 1), conv2d_0_pad_mode='pad')
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module2_1 = Module2(conv2d_0_in_channels=16, conv2d_0_kernel_size=(1, 1), conv2d_0_padding=0, conv2d_0_pad_mode='valid')
        self.module2_2 = Module2(conv2d_0_in_channels=16, conv2d_0_kernel_size=(3, 3), conv2d_0_padding=(1, 1, 1, 1), conv2d_0_pad_mode='pad')
        self.conv2d_8 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module5_0 = Module5(conv2d_0_in_channels=16, conv2d_0_out_channels=64, conv2d_2_in_channels=16, conv2d_2_out_channels=64, module0_0_batchnorm2d_0_num_features=64, module0_0_conv2d_2_in_channels=64, module0_0_conv2d_2_out_channels=16, module0_0_conv2d_2_stride=(1, 1), module0_0_conv2d_4_in_channels=16, module0_0_conv2d_4_out_channels=16, module0_1_batchnorm2d_0_num_features=64, module0_1_conv2d_2_in_channels=64, module0_1_conv2d_2_out_channels=16, module0_1_conv2d_2_stride=(1, 1), module0_1_conv2d_4_in_channels=16, module0_1_conv2d_4_out_channels=16)
        self.conv2d_26 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module3_0 = Module3(conv2d_0_in_channels=64, conv2d_0_out_channels=128, module0_0_batchnorm2d_0_num_features=64, module0_0_conv2d_2_in_channels=64, module0_0_conv2d_2_out_channels=64, module0_0_conv2d_2_stride=(2, 2), module0_0_conv2d_4_in_channels=64, module0_0_conv2d_4_out_channels=64)
        self.module5_1 = Module5(conv2d_0_in_channels=64, conv2d_0_out_channels=128, conv2d_2_in_channels=64, conv2d_2_out_channels=128, module0_0_batchnorm2d_0_num_features=128, module0_0_conv2d_2_in_channels=128, module0_0_conv2d_2_out_channels=64, module0_0_conv2d_2_stride=(1, 1), module0_0_conv2d_4_in_channels=64, module0_0_conv2d_4_out_channels=64, module0_1_batchnorm2d_0_num_features=128, module0_1_conv2d_2_in_channels=128, module0_1_conv2d_2_out_channels=64, module0_1_conv2d_2_stride=(1, 1), module0_1_conv2d_4_in_channels=64, module0_1_conv2d_4_out_channels=64)
        self.conv2d_51 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module3_1 = Module3(conv2d_0_in_channels=128, conv2d_0_out_channels=256, module0_0_batchnorm2d_0_num_features=128, module0_0_conv2d_2_in_channels=128, module0_0_conv2d_2_out_channels=128, module0_0_conv2d_2_stride=(2, 2), module0_0_conv2d_4_in_channels=128, module0_0_conv2d_4_out_channels=128)
        self.module5_2 = Module5(conv2d_0_in_channels=128, conv2d_0_out_channels=256, conv2d_2_in_channels=128, conv2d_2_out_channels=256, module0_0_batchnorm2d_0_num_features=256, module0_0_conv2d_2_in_channels=256, module0_0_conv2d_2_out_channels=128, module0_0_conv2d_2_stride=(1, 1), module0_0_conv2d_4_in_channels=128, module0_0_conv2d_4_out_channels=128, module0_1_batchnorm2d_0_num_features=256, module0_1_conv2d_2_in_channels=256, module0_1_conv2d_2_out_channels=128, module0_1_conv2d_2_stride=(1, 1), module0_1_conv2d_4_in_channels=128, module0_1_conv2d_4_out_channels=128)
        self.batchnorm2d_76 = nn.BatchNorm2d(num_features=256, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_77 = nn.ReLU()
        self.pad_avgpool2d_78 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_78 = nn.AvgPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.transpose_79 = P.Transpose()
        self.flatten_80 = nn.Flatten()
        self.matmul_81_w = Parameter(Tensor(np.random.uniform(0, 1, (256, 100)).astype(np.float32)), name=None)
        self.add_82_bias = Parameter(Tensor(np.random.uniform(0, 1, (100,)).astype(np.float32)), name=None)
        self.softmax_83 = nn.Softmax(axis=(- 1))
        self.add = mindspore.ops.Add()
        self.matmul = mindspore.ops.MatMul()

    def construct(self, input_1):
        opt_transpose_0 = self.transpose_0(input_1, (0, 3, 1, 2))
        module2_0_opt = self.module2_0(opt_transpose_0)
        opt_conv2d_3 = self.conv2d_3(module2_0_opt)
        module2_1_opt = self.module2_1(module2_0_opt)
        module2_2_opt = self.module2_2(module2_1_opt)
        opt_conv2d_8 = self.conv2d_8(module2_2_opt)
        opt_add_9 = self.add(opt_conv2d_3, opt_conv2d_8)
        module5_0_opt = self.module5_0(opt_add_9)
        opt_conv2d_26 = self.conv2d_26(module5_0_opt)
        module3_0_opt = self.module3_0(module5_0_opt)
        opt_add_34 = self.add(opt_conv2d_26, module3_0_opt)
        module5_1_opt = self.module5_1(opt_add_34)
        opt_conv2d_51 = self.conv2d_51(module5_1_opt)
        module3_1_opt = self.module3_1(module5_1_opt)
        opt_add_59 = self.add(opt_conv2d_51, module3_1_opt)
        module5_2_opt = self.module5_2(opt_add_59)
        opt_batchnorm2d_76 = self.batchnorm2d_76(module5_2_opt)
        opt_relu_77 = self.relu_77(opt_batchnorm2d_76)
        opt_avgpool2d_78 = self.pad_avgpool2d_78(opt_relu_77)
        opt_avgpool2d_78 = self.avgpool2d_78(opt_avgpool2d_78)
        opt_transpose_79 = self.transpose_79(opt_avgpool2d_78, (0, 2, 3, 1))
        opt_flatten_80 = self.flatten_80(opt_transpose_79)
        opt_matmul_81 = self.matmul(opt_flatten_80, self.matmul_81_w)
        opt_add_82 = (opt_matmul_81 + self.add_82_bias)
        opt_softmax_83 = self.softmax_83(opt_add_82)
        return opt_softmax_83

class Module2_1(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_kernel_size, conv2d_0_padding, conv2d_0_pad_mode):
        super(Module2_1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=16, kernel_size=conv2d_0_kernel_size, stride=(1, 1), padding=conv2d_0_padding, pad_mode=conv2d_0_pad_mode, dilation=(1, 1), group=1, has_bias=True)
        self.relu_1 = nn.ReLU()
        self.addNode_0 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW')

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_addNode_0 = self.addNode_0(opt_relu_1)
        return opt_addNode_0
