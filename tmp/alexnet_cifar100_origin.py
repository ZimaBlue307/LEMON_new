
from mindspore import nn
from mindspore.ops import functional as F

class Module5Opt_2(nn.Cell):

    def __init__(self, global_vars):
        super(Module5Opt_2, self).__init__()
        self._handler = global_vars.get('handler')
        self.module0_0 = Module0Opt(global_vars.get('module0_0_args'))
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')
        self.module0_1 = Module0Opt(global_vars.get('module0_1_args'))
        self.conv2d_2 = getattr(self._handler, 'conv2d_2')
        self.add = getattr(self._handler, 'add')

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d = self.conv2d_0(module0_0_opt)
        opt_add = self.add(x, opt_conv2d)
        module0_1_opt = self.module0_1(opt_add)
        opt_conv2d_1 = self.conv2d_2(module0_1_opt)
        opt_add_1 = self.add(opt_add, opt_conv2d_1)
        return opt_add_1

class Module3Opt_1(nn.Cell):

    def __init__(self, global_vars):
        super(Module3Opt_1, self).__init__()
        self._handler = global_vars.get('handler')
        self.module0_0 = Module0Opt(global_vars.get('module0_0_args'))
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d = self.conv2d_0(module0_0_opt)
        return opt_conv2d

class Module5Opt_1(nn.Cell):

    def __init__(self, global_vars):
        super(Module5Opt_1, self).__init__()
        self._handler = global_vars.get('handler')
        self.module0_0 = Module0Opt(global_vars.get('module0_0_args'))
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')
        self.module0_1 = Module0Opt(global_vars.get('module0_1_args'))
        self.conv2d_2 = getattr(self._handler, 'conv2d_2')
        self.add = getattr(self._handler, 'add')

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d = self.conv2d_0(module0_0_opt)
        opt_add = self.add(x, opt_conv2d)
        module0_1_opt = self.module0_1(opt_add)
        opt_conv2d_1 = self.conv2d_2(module0_1_opt)
        opt_add_1 = self.add(opt_add, opt_conv2d_1)
        return opt_add_1

class Module3Opt(nn.Cell):

    def __init__(self, global_vars):
        super(Module3Opt, self).__init__()
        self._handler = global_vars.get('handler')
        self.module0_0 = Module0Opt(global_vars.get('module0_0_args'))
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d = self.conv2d_0(module0_0_opt)
        return opt_conv2d

class Module0Opt(nn.Cell):

    def __init__(self, global_vars):
        super(Module0Opt, self).__init__()
        self._handler = global_vars.get('handler')
        self.batchnorm2d_0 = getattr(self._handler, 'batchnorm2d_0')
        self.relu_1 = getattr(self._handler, 'relu_1')
        self.conv2d_2 = getattr(self._handler, 'conv2d_2')
        self.relu_3 = getattr(self._handler, 'relu_3')
        self.conv2d_4 = getattr(self._handler, 'conv2d_4')
        self.relu_5 = getattr(self._handler, 'relu_5')

    def construct(self, x):
        opt_batchnorm2d = self.batchnorm2d_0(x)
        opt_relu = self.relu_1(opt_batchnorm2d)
        opt_conv2d = self.conv2d_2(opt_relu)
        opt_relu_1 = self.relu_3(opt_conv2d)
        opt_conv2d_1 = self.conv2d_4(opt_relu_1)
        opt_relu_2 = self.relu_5(opt_conv2d_1)
        return opt_relu_2

class Module5Opt(nn.Cell):

    def __init__(self, global_vars):
        super(Module5Opt, self).__init__()
        self._handler = global_vars.get('handler')
        self.module0_0 = Module0Opt(global_vars.get('module0_0_args'))
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')
        self.module0_1 = Module0Opt(global_vars.get('module0_1_args'))
        self.conv2d_2 = getattr(self._handler, 'conv2d_2')
        self.add = getattr(self._handler, 'add')

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d = self.conv2d_0(module0_0_opt)
        opt_add = self.add(x, opt_conv2d)
        module0_1_opt = self.module0_1(opt_add)
        opt_conv2d_1 = self.conv2d_2(module0_1_opt)
        opt_add_1 = self.add(opt_add, opt_conv2d_1)
        return opt_add_1

class Module2Opt(nn.Cell):

    def __init__(self, global_vars):
        super(Module2Opt, self).__init__()
        self._handler = global_vars.get('handler')
        self.conv2d_0 = getattr(self._handler, 'conv2d_0')
        self.relu_1 = getattr(self._handler, 'relu_1')

    def construct(self, x):
        opt_conv2d = self.conv2d_0(x)
        opt_relu = self.relu_1(opt_conv2d)
        return opt_relu

class MindSporeModelOpt(nn.Cell):

    def __init__(self, global_vars):
        super(MindSporeModelOpt, self).__init__()
        self._handler = global_vars.get('handler')
        self.transpose_0 = getattr(self._handler, 'transpose_0')
        self.module2_0 = Module2Opt(global_vars.get('module2_0_args'))
        self.conv2d_3 = getattr(self._handler, 'conv2d_3')
        self.module2_1 = Module2Opt(global_vars.get('module2_1_args'))
        self.module2_2 = Module2Opt(global_vars.get('module2_2_args'))
        self.conv2d_8 = getattr(self._handler, 'conv2d_8')
        self.module5_0 = Module5Opt(global_vars.get('module5_0_args'))
        self.conv2d_26 = getattr(self._handler, 'conv2d_26')
        self.module3_0 = Module3Opt(global_vars.get('module3_0_args'))
        self.module5_1 = Module5Opt_1(global_vars.get('module5_1_args'))
        self.conv2d_51 = getattr(self._handler, 'conv2d_51')
        self.module3_1 = Module3Opt_1(global_vars.get('module3_1_args'))
        self.module5_2 = Module5Opt_2(global_vars.get('module5_2_args'))
        self.batchnorm2d_76 = getattr(self._handler, 'batchnorm2d_76')
        self.relu_77 = getattr(self._handler, 'relu_77')
        self.pad_avgpool2d_78 = getattr(self._handler, 'pad_avgpool2d_78')
        self.avgpool2d_78 = getattr(self._handler, 'avgpool2d_78')
        self.transpose_79 = getattr(self._handler, 'transpose_79')
        self.flatten_80 = getattr(self._handler, 'flatten_80')
        self.matmul_81_w = getattr(self._handler, 'matmul_81_w')
        self.add_82_bias = getattr(self._handler, 'add_82_bias')
        self.softmax_83 = getattr(self._handler, 'softmax_83')
        self.add = getattr(self._handler, 'add')
        self.matmul = getattr(self._handler, 'matmul')

    def construct(self, input_1):
        tuple_var = (0, 3, 1, 2)
        opt_transpose = self.transpose_0(input, tuple_var)
        module2_0_opt = self.module2_0(opt_transpose)
        opt_conv2d = self.conv2d_3(module2_0_opt)
        module2_1_opt = self.module2_1(module2_0_opt)
        module2_2_opt = self.module2_2(module2_1_opt)
        opt_conv2d_1 = self.conv2d_8(module2_2_opt)
        opt_add = self.add(opt_conv2d, opt_conv2d_1)
        module5_0_opt = self.module5_0(opt_add)
        opt_conv2d_2 = self.conv2d_26(module5_0_opt)
        module3_0_opt = self.module3_0(module5_0_opt)
        opt_add_1 = self.add(opt_conv2d_2, module3_0_opt)
        module5_1_opt = self.module5_1(opt_add_1)
        opt_conv2d_3 = self.conv2d_51(module5_1_opt)
        module3_1_opt = self.module3_1(module5_1_opt)
        opt_add_2 = self.add(opt_conv2d_3, module3_1_opt)
        module5_2_opt = self.module5_2(opt_add_2)
        opt_batchnorm2d = self.batchnorm2d_76(module5_2_opt)
        opt_relu = self.relu_77(opt_batchnorm2d)
        opt_avgpool2d = self.pad_avgpool2d_78(opt_relu)
        opt_avgpool2d_1 = self.avgpool2d_78(opt_avgpool2d)
        tuple_var_1 = (0, 2, 3, 1)
        opt_transpose_1 = self.transpose_79(opt_avgpool2d_1, tuple_var_1)
        opt_flatten = self.flatten_80(opt_transpose_1)
        self_matmul_81_w = self.matmul_81_w
        opt_matmul = self.matmul(opt_flatten, self_matmul_81_w)
        self_add_82_bias = self.add_82_bias
        opt_add_3 = F.add(opt_matmul, self_add_82_bias)
        opt_softmax = self.softmax_83(opt_add_3)
        return opt_softmax
