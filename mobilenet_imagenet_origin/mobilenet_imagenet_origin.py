import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, conv2d_0_group, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_2_kernel_size, conv2d_2_stride, conv2d_2_padding, conv2d_2_pad_mode, conv2d_2_group,
                 conv2d_4_in_channels, conv2d_4_out_channels, conv2d_4_kernel_size, conv2d_4_stride, conv2d_4_padding,
                 conv2d_4_pad_mode, conv2d_4_group):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_0_group,
                                  has_bias=True)
        self.clip_by_value_1_min = 0.0
        self.clip_by_value_1_max = 6.0
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=conv2d_2_kernel_size,
                                  stride=conv2d_2_stride,
                                  padding=conv2d_2_padding,
                                  pad_mode=conv2d_2_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_2_group,
                                  has_bias=True)
        self.clip_by_value_3_min = 0.0
        self.clip_by_value_3_max = 6.0
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=conv2d_4_kernel_size,
                                  stride=conv2d_4_stride,
                                  padding=conv2d_4_padding,
                                  pad_mode=conv2d_4_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_4_group,
                                  has_bias=True)
        self.clip_by_value_5_min = 0.0
        self.clip_by_value_5_max = 6.0

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_clip_by_value_1 = P.clip_by_value(opt_conv2d_0, self.clip_by_value_1_min, self.clip_by_value_1_max)
        opt_conv2d_2 = self.conv2d_2(opt_clip_by_value_1)
        opt_clip_by_value_3 = P.clip_by_value(opt_conv2d_2, self.clip_by_value_3_min, self.clip_by_value_3_max)
        opt_conv2d_4 = self.conv2d_4(opt_clip_by_value_3)
        opt_clip_by_value_5 = P.clip_by_value(opt_conv2d_4, self.clip_by_value_5_min, self.clip_by_value_5_max)
        return opt_clip_by_value_5


class Module1(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size,
                 module0_0_conv2d_0_stride, module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode,
                 module0_0_conv2d_0_group, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_kernel_size, module0_0_conv2d_2_stride, module0_0_conv2d_2_padding,
                 module0_0_conv2d_2_pad_mode, module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_4_kernel_size, module0_0_conv2d_4_stride,
                 module0_0_conv2d_4_padding, module0_0_conv2d_4_pad_mode, module0_0_conv2d_4_group,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_0_kernel_size,
                 module0_1_conv2d_0_stride, module0_1_conv2d_0_padding, module0_1_conv2d_0_pad_mode,
                 module0_1_conv2d_0_group, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_kernel_size, module0_1_conv2d_2_stride, module0_1_conv2d_2_padding,
                 module0_1_conv2d_2_pad_mode, module0_1_conv2d_2_group, module0_1_conv2d_4_in_channels,
                 module0_1_conv2d_4_out_channels, module0_1_conv2d_4_kernel_size, module0_1_conv2d_4_stride,
                 module0_1_conv2d_4_padding, module0_1_conv2d_4_pad_mode, module0_1_conv2d_4_group):
        super(Module1, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module0_0_conv2d_0_group,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_kernel_size=module0_0_conv2d_2_kernel_size,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_2_padding=module0_0_conv2d_2_padding,
                                 conv2d_2_pad_mode=module0_0_conv2d_2_pad_mode,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels,
                                 conv2d_4_kernel_size=module0_0_conv2d_4_kernel_size,
                                 conv2d_4_stride=module0_0_conv2d_4_stride,
                                 conv2d_4_padding=module0_0_conv2d_4_padding,
                                 conv2d_4_pad_mode=module0_0_conv2d_4_pad_mode,
                                 conv2d_4_group=module0_0_conv2d_4_group)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_1_conv2d_0_stride,
                                 conv2d_0_padding=module0_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module0_1_conv2d_0_group,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_kernel_size=module0_1_conv2d_2_kernel_size,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_2_padding=module0_1_conv2d_2_padding,
                                 conv2d_2_pad_mode=module0_1_conv2d_2_pad_mode,
                                 conv2d_2_group=module0_1_conv2d_2_group,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels,
                                 conv2d_4_kernel_size=module0_1_conv2d_4_kernel_size,
                                 conv2d_4_stride=module0_1_conv2d_4_stride,
                                 conv2d_4_padding=module0_1_conv2d_4_padding,
                                 conv2d_4_pad_mode=module0_1_conv2d_4_pad_mode,
                                 conv2d_4_group=module0_1_conv2d_4_group)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module1_0 = Module1(module0_0_conv2d_0_in_channels=3,
                                 module0_0_conv2d_0_out_channels=32,
                                 module0_0_conv2d_0_kernel_size=(3, 3),
                                 module0_0_conv2d_0_stride=(2, 2),
                                 module0_0_conv2d_0_padding=(0, 1, 0, 1),
                                 module0_0_conv2d_0_pad_mode="pad",
                                 module0_0_conv2d_0_group=1,
                                 module0_0_conv2d_2_in_channels=32,
                                 module0_0_conv2d_2_out_channels=32,
                                 module0_0_conv2d_2_kernel_size=(3, 3),
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_padding=(1, 1, 1, 1),
                                 module0_0_conv2d_2_pad_mode="pad",
                                 module0_0_conv2d_2_group=32,
                                 module0_0_conv2d_4_in_channels=32,
                                 module0_0_conv2d_4_out_channels=64,
                                 module0_0_conv2d_4_kernel_size=(1, 1),
                                 module0_0_conv2d_4_stride=(1, 1),
                                 module0_0_conv2d_4_padding=0,
                                 module0_0_conv2d_4_pad_mode="valid",
                                 module0_0_conv2d_4_group=1,
                                 module0_1_conv2d_0_in_channels=64,
                                 module0_1_conv2d_0_out_channels=64,
                                 module0_1_conv2d_0_kernel_size=(3, 3),
                                 module0_1_conv2d_0_stride=(2, 2),
                                 module0_1_conv2d_0_padding=(0, 1, 0, 1),
                                 module0_1_conv2d_0_pad_mode="pad",
                                 module0_1_conv2d_0_group=64,
                                 module0_1_conv2d_2_in_channels=64,
                                 module0_1_conv2d_2_out_channels=128,
                                 module0_1_conv2d_2_kernel_size=(1, 1),
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_padding=0,
                                 module0_1_conv2d_2_pad_mode="valid",
                                 module0_1_conv2d_2_group=1,
                                 module0_1_conv2d_4_in_channels=128,
                                 module0_1_conv2d_4_out_channels=128,
                                 module0_1_conv2d_4_kernel_size=(3, 3),
                                 module0_1_conv2d_4_stride=(1, 1),
                                 module0_1_conv2d_4_padding=(1, 1, 1, 1),
                                 module0_1_conv2d_4_pad_mode="pad",
                                 module0_1_conv2d_4_group=128)
        self.module1_1 = Module1(module0_0_conv2d_0_in_channels=128,
                                 module0_0_conv2d_0_out_channels=128,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid",
                                 module0_0_conv2d_0_group=1,
                                 module0_0_conv2d_2_in_channels=128,
                                 module0_0_conv2d_2_out_channels=128,
                                 module0_0_conv2d_2_kernel_size=(3, 3),
                                 module0_0_conv2d_2_stride=(2, 2),
                                 module0_0_conv2d_2_padding=(0, 1, 0, 1),
                                 module0_0_conv2d_2_pad_mode="pad",
                                 module0_0_conv2d_2_group=128,
                                 module0_0_conv2d_4_in_channels=128,
                                 module0_0_conv2d_4_out_channels=256,
                                 module0_0_conv2d_4_kernel_size=(1, 1),
                                 module0_0_conv2d_4_stride=(1, 1),
                                 module0_0_conv2d_4_padding=0,
                                 module0_0_conv2d_4_pad_mode="valid",
                                 module0_0_conv2d_4_group=1,
                                 module0_1_conv2d_0_in_channels=256,
                                 module0_1_conv2d_0_out_channels=256,
                                 module0_1_conv2d_0_kernel_size=(3, 3),
                                 module0_1_conv2d_0_stride=(1, 1),
                                 module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module0_1_conv2d_0_pad_mode="pad",
                                 module0_1_conv2d_0_group=256,
                                 module0_1_conv2d_2_in_channels=256,
                                 module0_1_conv2d_2_out_channels=256,
                                 module0_1_conv2d_2_kernel_size=(1, 1),
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_padding=0,
                                 module0_1_conv2d_2_pad_mode="valid",
                                 module0_1_conv2d_2_group=1,
                                 module0_1_conv2d_4_in_channels=256,
                                 module0_1_conv2d_4_out_channels=256,
                                 module0_1_conv2d_4_kernel_size=(3, 3),
                                 module0_1_conv2d_4_stride=(2, 2),
                                 module0_1_conv2d_4_padding=(0, 1, 0, 1),
                                 module0_1_conv2d_4_pad_mode="pad",
                                 module0_1_conv2d_4_group=256)
        self.module1_2 = Module1(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid",
                                 module0_0_conv2d_0_group=1,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_2_kernel_size=(3, 3),
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_padding=(1, 1, 1, 1),
                                 module0_0_conv2d_2_pad_mode="pad",
                                 module0_0_conv2d_2_group=512,
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=512,
                                 module0_0_conv2d_4_kernel_size=(1, 1),
                                 module0_0_conv2d_4_stride=(1, 1),
                                 module0_0_conv2d_4_padding=0,
                                 module0_0_conv2d_4_pad_mode="valid",
                                 module0_0_conv2d_4_group=1,
                                 module0_1_conv2d_0_in_channels=512,
                                 module0_1_conv2d_0_out_channels=512,
                                 module0_1_conv2d_0_kernel_size=(3, 3),
                                 module0_1_conv2d_0_stride=(1, 1),
                                 module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module0_1_conv2d_0_pad_mode="pad",
                                 module0_1_conv2d_0_group=512,
                                 module0_1_conv2d_2_in_channels=512,
                                 module0_1_conv2d_2_out_channels=512,
                                 module0_1_conv2d_2_kernel_size=(1, 1),
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_padding=0,
                                 module0_1_conv2d_2_pad_mode="valid",
                                 module0_1_conv2d_2_group=1,
                                 module0_1_conv2d_4_in_channels=512,
                                 module0_1_conv2d_4_out_channels=512,
                                 module0_1_conv2d_4_kernel_size=(3, 3),
                                 module0_1_conv2d_4_stride=(1, 1),
                                 module0_1_conv2d_4_padding=(1, 1, 1, 1),
                                 module0_1_conv2d_4_pad_mode="pad",
                                 module0_1_conv2d_4_group=512)
        self.module1_3 = Module1(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid",
                                 module0_0_conv2d_0_group=1,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_2_kernel_size=(3, 3),
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_2_padding=(1, 1, 1, 1),
                                 module0_0_conv2d_2_pad_mode="pad",
                                 module0_0_conv2d_2_group=512,
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=512,
                                 module0_0_conv2d_4_kernel_size=(1, 1),
                                 module0_0_conv2d_4_stride=(1, 1),
                                 module0_0_conv2d_4_padding=0,
                                 module0_0_conv2d_4_pad_mode="valid",
                                 module0_0_conv2d_4_group=1,
                                 module0_1_conv2d_0_in_channels=512,
                                 module0_1_conv2d_0_out_channels=512,
                                 module0_1_conv2d_0_kernel_size=(3, 3),
                                 module0_1_conv2d_0_stride=(1, 1),
                                 module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module0_1_conv2d_0_pad_mode="pad",
                                 module0_1_conv2d_0_group=512,
                                 module0_1_conv2d_2_in_channels=512,
                                 module0_1_conv2d_2_out_channels=512,
                                 module0_1_conv2d_2_kernel_size=(1, 1),
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_2_padding=0,
                                 module0_1_conv2d_2_pad_mode="valid",
                                 module0_1_conv2d_2_group=1,
                                 module0_1_conv2d_4_in_channels=512,
                                 module0_1_conv2d_4_out_channels=512,
                                 module0_1_conv2d_4_kernel_size=(3, 3),
                                 module0_1_conv2d_4_stride=(2, 2),
                                 module0_1_conv2d_4_padding=(0, 1, 0, 1),
                                 module0_1_conv2d_4_pad_mode="pad",
                                 module0_1_conv2d_4_group=512)
        self.module0_0 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=1024,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid",
                                 conv2d_0_group=1,
                                 conv2d_2_in_channels=1024,
                                 conv2d_2_out_channels=1024,
                                 conv2d_2_kernel_size=(3, 3),
                                 conv2d_2_stride=(1, 1),
                                 conv2d_2_padding=(1, 1, 1, 1),
                                 conv2d_2_pad_mode="pad",
                                 conv2d_2_group=1024,
                                 conv2d_4_in_channels=1024,
                                 conv2d_4_out_channels=1024,
                                 conv2d_4_kernel_size=(1, 1),
                                 conv2d_4_stride=(1, 1),
                                 conv2d_4_padding=0,
                                 conv2d_4_pad_mode="valid",
                                 conv2d_4_group=1)
        self.avgpool2d_55 = nn.AvgPool2d(kernel_size=(7, 7))
        self.transpose_56 = P.Transpose()
        self.reshape_57 = P.Reshape()
        self.reshape_57_shape = tuple([1, 1024])
        self.reshape_58 = P.Reshape()
        self.reshape_58_shape = tuple([1, 1024, 1, 1])
        self.conv2d_59 = nn.Conv2d(in_channels=1024,
                                   out_channels=1000,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.transpose_60 = P.Transpose()
        self.softmax_61 = nn.Softmax(axis=-1)
        self.reshape_62 = P.Reshape()
        self.reshape_62_shape = tuple([1, 1000])

    def construct(self, input_3):
        opt_transpose_0 = self.transpose_0(input_3, (0, 3, 1, 2))
        module1_0_opt = self.module1_0(opt_transpose_0)
        module1_1_opt = self.module1_1(module1_0_opt)
        module1_2_opt = self.module1_2(module1_1_opt)
        module1_3_opt = self.module1_3(module1_2_opt)
        module0_0_opt = self.module0_0(module1_3_opt)
        opt_avgpool2d_55 = self.avgpool2d_55(module0_0_opt)
        opt_transpose_56 = self.transpose_56(opt_avgpool2d_55, (0, 2, 3, 1))
        opt_reshape_57 = self.reshape_57(opt_transpose_56, self.reshape_57_shape)
        opt_reshape_58 = self.reshape_58(opt_reshape_57, self.reshape_58_shape)
        opt_conv2d_59 = self.conv2d_59(opt_reshape_58)
        opt_transpose_60 = self.transpose_60(opt_conv2d_59, (0, 2, 3, 1))
        opt_softmax_61 = self.softmax_61(opt_transpose_60)
        opt_reshape_62 = self.reshape_62(opt_softmax_61, self.reshape_62_shape)
        return opt_reshape_62
