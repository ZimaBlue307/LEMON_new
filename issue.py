import mindspore
import pickle
from mindspore import nn, rewrite
from mindspore.nn import Dense, Conv2d, ReLU


if __name__ == "__main__":
    cell = Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2), pad_mode="valid",
                            dilation=(1, 1), has_bias=True)
    with open('save_tmp.pkl', 'wb') as f:
        pickle.dump(cell, f, pickle.HIGHEST_PROTOCOL)

    with open('save_tmp.pkl', 'rb') as fr:
        handler = pickle.load(fr)
