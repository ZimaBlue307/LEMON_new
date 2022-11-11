#model_name,dataset_name,mut_model,entire_model_name = "resnet20","cifar100", "mut_model", "resnet20_cifar100_ori_WS32"
import sys
sys.path.append("..")
sys.path.append("..")
from lemon_outputs.WS2_only.resnet20_cifar100.mut_model.resnet20_cifar100_ori_WS32.resnet20_cifar100_ori_WS32 import MindSporeModel
import mindspore
import numpy as np
def auto_import_msmodel():
	auto_model = MindSporeModel()
	# param_dict = mindspore.load_checkpoint("/home/lemon_proj/lyh/LEMON_new/lemon_outputs/WS2_only/resnet20_cifar100/mut_model/resnet20_cifar100_ori_WS32/resnet20_cifar100_ori_WS32.ckpt")
	# mindspore.load_param_into_net(auto_model, param_dict)
	return auto_model