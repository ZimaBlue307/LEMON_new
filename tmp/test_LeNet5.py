
from mindspore import nn

class LeNet5Opt(nn.Cell):
    '\n    LeNet-5网络结构\n    '

    def __init__(self, global_vars):
        super(LeNet5Opt, self).__init__()
        self._handler = global_vars.get('handler')
        self.conv1 = getattr(self._handler, 'conv1')
        self.conv2 = getattr(self._handler, 'conv2')
        self.fc1 = getattr(self._handler, 'fc1')
        self.fc2 = getattr(self._handler, 'fc2')
        self.fc3 = getattr(self._handler, 'fc3')
        self.relu = getattr(self._handler, 'relu')
        self.max_pool2d = getattr(self._handler, 'max_pool2d')
        self.flatten = getattr(self._handler, 'flatten')

    def construct(self, x):
        x_1 = self.conv1(x)
        x_2 = self.relu(x_1)
        x_3 = self.max_pool2d(x_2)
        x_4 = self.conv2(x_3)
        x_5 = self.relu(x_4)
        x_6 = self.max_pool2d(x_5)
        x_7 = self.flatten(x_6)
        x_8 = self.fc1(x_7)
        x_9 = self.relu(x_8)
        x_10 = self.fc2(x_9)
        x_11 = self.relu(x_10)
        x_12 = self.fc3(x_11)
        return x_12
