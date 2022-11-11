
import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter
import json
save = list()

class Module2(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_kernel_size, conv2d_0_padding, conv2d_0_pad_mode):
        super(Module2, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=16, kernel_size=conv2d_0_kernel_size, stride=(1, 1), padding=conv2d_0_padding, pad_mode=conv2d_0_pad_mode, dilation=(1, 1), group=1, has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        global save
        opt_conv2d_0 = self.conv2d_0(x)
        save.append(['opt_conv2d_0', opt_conv2d_0.shape])
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        save.append(['opt_relu_1', opt_relu_1.shape])
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
        global save
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        save.append(['opt_batchnorm2d_0', opt_batchnorm2d_0.shape])
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        save.append(['opt_relu_1', opt_relu_1.shape])
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        save.append(['opt_conv2d_2', opt_conv2d_2.shape])
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        save.append(['opt_relu_3', opt_relu_3.shape])
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        save.append(['opt_conv2d_4', opt_conv2d_4.shape])
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        save.append(['opt_relu_5', opt_relu_5.shape])
        return opt_relu_5

class Module5(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels, module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module5, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_0_conv2d_2_in_channels, conv2d_2_out_channels=module0_0_conv2d_2_out_channels, conv2d_2_stride=module0_0_conv2d_2_stride, conv2d_4_in_channels=module0_0_conv2d_4_in_channels, conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=conv2d_0_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_1_conv2d_2_in_channels, conv2d_2_out_channels=module0_1_conv2d_2_out_channels, conv2d_2_stride=module0_1_conv2d_2_stride, conv2d_4_in_channels=module0_1_conv2d_4_in_channels, conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels, out_channels=conv2d_2_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)

    def construct(self, x):
        global save
        save.append(['module0_0_opt', 'module0_0_opt start'])
        module0_0_opt = self.module0_0.construct(x)
        save.append(['module0_0_opt', 'module0_0_opt end'])
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        save.append(['opt_conv2d_0', opt_conv2d_0.shape])
        opt_add_1 = P.Add()(x, opt_conv2d_0)
        save.append(['opt_add_1', opt_add_1.shape])
        save.append(['module0_1_opt', 'module0_1_opt start'])
        module0_1_opt = self.module0_1.construct(opt_add_1)
        save.append(['module0_1_opt', 'module0_1_opt end'])
        opt_conv2d_2 = self.conv2d_2(module0_1_opt)
        save.append(['opt_conv2d_2', opt_conv2d_2.shape])
        opt_add_3 = P.Add()(opt_add_1, opt_conv2d_2)
        save.append(['opt_add_3', opt_add_3.shape])
        return opt_add_3

class Module3(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels):
        super(Module3, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features, conv2d_2_in_channels=module0_0_conv2d_2_in_channels, conv2d_2_out_channels=module0_0_conv2d_2_out_channels, conv2d_2_stride=module0_0_conv2d_2_stride, conv2d_4_in_channels=module0_0_conv2d_4_in_channels, conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels, out_channels=conv2d_0_out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True)

    def construct(self, x):
        global save
        save.append(['module0_0_opt', 'module0_0_opt start'])
        module0_0_opt = self.module0_0.construct(x)
        save.append(['module0_0_opt', 'module0_0_opt end'])
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        save.append(['opt_conv2d_0', opt_conv2d_0.shape])
        return opt_conv2d_0

class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module2_0 = Module2(conv2d_0_in_channels=3, conv2d_0_kernel_size=(3, 3), conv2d_0_padding=(1, 1, 1, 1), conv2d_0_pad_mode='pad')
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

    def construct(self, input_1):
        global save
        opt_transpose_0 = self.transpose_0(input_1, (0, 3, 1, 2))
        save.append(['opt_transpose_0', opt_transpose_0.shape])
        save.append(['module2_0_opt', 'module2_0_opt start'])
        module2_0_opt = self.module2_0.construct(opt_transpose_0)
        save.append(['module2_0_opt', 'module2_0_opt end'])
        opt_conv2d_3 = self.conv2d_3(module2_0_opt)
        save.append(['opt_conv2d_3', opt_conv2d_3.shape])
        save.append(['module2_1_opt', 'module2_1_opt start'])
        module2_1_opt = self.module2_1.construct(module2_0_opt)
        save.append(['module2_1_opt', 'module2_1_opt end'])
        save.append(['module2_2_opt', 'module2_2_opt start'])
        module2_2_opt = self.module2_2.construct(module2_1_opt)
        save.append(['module2_2_opt', 'module2_2_opt end'])
        opt_conv2d_8 = self.conv2d_8(module2_2_opt)
        save.append(['opt_conv2d_8', opt_conv2d_8.shape])
        opt_add_9 = P.Add()(opt_conv2d_3, opt_conv2d_8)
        save.append(['opt_add_9', opt_add_9.shape])
        save.append(['module5_0_opt', 'module5_0_opt start'])
        module5_0_opt = self.module5_0.construct(opt_add_9)
        save.append(['module5_0_opt', 'module5_0_opt end'])
        opt_conv2d_26 = self.conv2d_26(module5_0_opt)
        save.append(['opt_conv2d_26', opt_conv2d_26.shape])
        save.append(['module3_0_opt', 'module3_0_opt start'])
        module3_0_opt = self.module3_0.construct(module5_0_opt)
        save.append(['module3_0_opt', 'module3_0_opt end'])
        opt_add_34 = P.Add()(opt_conv2d_26, module3_0_opt)
        save.append(['opt_add_34', opt_add_34.shape])
        save.append(['module5_1_opt', 'module5_1_opt start'])
        module5_1_opt = self.module5_1.construct(opt_add_34)
        save.append(['module5_1_opt', 'module5_1_opt end'])
        opt_conv2d_51 = self.conv2d_51(module5_1_opt)
        save.append(['opt_conv2d_51', opt_conv2d_51.shape])
        save.append(['module3_1_opt', 'module3_1_opt start'])
        module3_1_opt = self.module3_1.construct(module5_1_opt)
        save.append(['module3_1_opt', 'module3_1_opt end'])
        opt_add_59 = P.Add()(opt_conv2d_51, module3_1_opt)
        save.append(['opt_add_59', opt_add_59.shape])
        save.append(['module5_2_opt', 'module5_2_opt start'])
        module5_2_opt = self.module5_2.construct(opt_add_59)
        save.append(['module5_2_opt', 'module5_2_opt end'])
        opt_batchnorm2d_76 = self.batchnorm2d_76(module5_2_opt)
        save.append(['opt_batchnorm2d_76', opt_batchnorm2d_76.shape])
        opt_relu_77 = self.relu_77(opt_batchnorm2d_76)
        save.append(['opt_relu_77', opt_relu_77.shape])
        opt_avgpool2d_78 = self.pad_avgpool2d_78(opt_relu_77)
        save.append(['opt_avgpool2d_78', opt_avgpool2d_78.shape])
        opt_avgpool2d_78 = self.avgpool2d_78(opt_avgpool2d_78)
        save.append(['opt_avgpool2d_78', opt_avgpool2d_78.shape])
        opt_transpose_79 = self.transpose_79(opt_avgpool2d_78, (0, 2, 3, 1))
        save.append(['opt_transpose_79', opt_transpose_79.shape])
        opt_flatten_80 = self.flatten_80(opt_transpose_79)
        save.append(['opt_flatten_80', opt_flatten_80.shape])
        opt_matmul_81 = P.matmul(opt_flatten_80, self.matmul_81_w)
        save.append(['opt_matmul_81', opt_matmul_81.shape])
        opt_add_82 = (opt_matmul_81 + self.add_82_bias)
        opt_softmax_83 = self.softmax_83(opt_add_82)
        save.append(['opt_softmax_83', opt_softmax_83.shape])
        with open('shape_tmp.json', 'w') as f:
            json.dump(save, f)
        return opt_softmax_83
