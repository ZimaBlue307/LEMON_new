import mindspore
from lemon_outputs.resnet20_cifar100.mut_model.resnet20_cifar100_origin0.resnet20_cifar100_origin0 import MindSporeModel


if __name__ == "__main__":
    resnet20_cifar100 = MindSporeModel()
    param_dict = mindspore.load_checkpoint(f'origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt')
    mindspore.load_param_into_net(resnet20_cifar100, param_dict)
    