# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Alexnet."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
#
# def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                      has_bias=has_bias, pad_mode=pad_mode)

def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)

class DataNormTranspose(nn.Cell):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respectively.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """
    def __init__(self):
        super(DataNormTranspose, self).__init__()
        self.mean = Tensor(np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape((1, 1, 1, 3)), mstype.float32)
        self.std = Tensor(np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape((1, 1, 1, 3)), mstype.float32)

    def construct(self, x):
        x = (x - self.mean) / self.std
        x = F.transpose(x, (0, 3, 1, 2))
        return x

class MindSporeModel(nn.Cell):
    """
    Alexnet
    """
    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True, off_load=False):
        super(MindSporeModel, self).__init__()
        self.off_load = off_load
        if self.off_load is True:
            self.data_trans = DataNormTranspose()
        # self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=11, stride=4, pad_mode='same', has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, pad_mode='same', has_bias=True)
        # self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, pad_mode="same", has_bias=True)
        # self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, pad_mode="same", has_bias=True)
        # self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode="same", has_bias=True)
        # self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            # self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc1 = nn.Dense(in_channels=6*6*256, out_channels=4096, has_bias=True)
            # self.fc2 = fc_with_initialize(4096, 4096)
            self.fc2 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
            # self.fc3 = fc_with_initialize(4096, num_classes)
            self.fc3 = nn.Dense(in_channels=4096, out_channels=num_classes, has_bias=True)
            self.dropout = nn.Dropout(dropout_ratio)

    def construct(self, x):
        """define network"""
        # if self.off_load is True:
        #     x = self.data_trans(x)
        conv1_opt = self.conv1(x)
        relu_1_opt = self.relu(conv1_opt)
        max_pool2d_1_opt = self.max_pool2d(relu_1_opt)
        conv2_opt = self.conv2(max_pool2d_1_opt)
        relu_2_opt = self.relu(conv2_opt)
        max_pool2d_2_opt = self.max_pool2d(relu_2_opt)
        conv3_opt = self.conv3(max_pool2d_2_opt)
        relu_3_opt = self.relu(conv3_opt)
        conv4_opt = self.conv4(relu_3_opt)
        relu_4_opt = self.relu(conv4_opt)
        conv5_opt = self.conv5(relu_4_opt)
        relu_5_opt = self.relu(conv5_opt)
        max_pool2d_3_opt = self.max_pool2d(relu_5_opt)
        if not self.include_top:
            return max_pool2d_3_opt
        flatten_opt = self.flatten(max_pool2d_3_opt)
        fc1_opt = self.fc1(flatten_opt)
        relu_6_opt = self.relu(fc1_opt)
        dropout_1_opt = self.dropout(relu_6_opt)
        fc2_opt = self.fc2(dropout_1_opt)
        relu_7_opt = self.relu(fc2_opt)
        dropout_2_opt = self.dropout(relu_7_opt)
        fc3_opt = self.fc3(dropout_2_opt)
        return fc3_opt