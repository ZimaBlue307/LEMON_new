# -*-coding:UTF-8-*-
"""get prediction for each backend
"""
import sys
import os
import redis
import pickle
import argparse
import configparser
from scripts.tools.utils import DataUtils
from scripts.logger.lemon_logger import Logger
import warnings
import mindspore
from mindspore import nn
import tempfile

main_logger = Logger()


def custom_objects():

    class No_Activation(nn.Cell):
        def __init__(self):
            super(No_Activation, self).__init__()
        def construct(self, x):
            result = x
            return result

    class leakyrelu_layer(nn.Cell):
        def __init__(self):
            super(leakyrelu_layer, self).__init__()
            self.leakyrelu = nn.LeakyReLU(alpha = 0.01)
        def construct(self, x):
            result = self.leakyrelu(x)
            return result

    objects = {}
    no_act_class = No_Activation()
    leakyrelu_class = leakyrelu_layer()
    objects['no_activation'] = no_act_class
    objects['leakyrelu'] = leakyrelu_class
    return objects

def get_cls_through_file(model_path, ckpt_name):
    _opt_cls_name = get_opt_cls_name()
    tmp_module_path = model_path
    tmp_module_file = ckpt_name + ".py"
    tmp_module_name = tmp_module_file[:-3]
    sys.path.append(tmp_module_path)
    tmp_module = __import__(tmp_module_name) 
    network_cls = getattr(tmp_module, _opt_cls_name)
    if network_cls is None:
            raise RuntimeError("Can not find network class: ", _opt_cls_name)
    return network_cls #是一个MindSporeModel的class
    
def get_opt_cls_name():
    mindsporemodel_var = "MindSporeModel"
    return mindsporemodel_var

#for example, model_path == /home/lemon_proj/lyh/LEMON_new/lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_origin0
def _get_prediction(bk, pred_dataset, model_path, batch_num):
    """
    Get prediction of models on different backends
    assuming all model structures can be imported through the statement model = MindsporeModel(), 
    which means that the right side of the equal sign has always been MindsporeModel
    """
    # from scripts.mutation.hyr_import_ms_model import auto_generate_import_model_script
    # auto_generate_import_model_script(model_path)
    # import scripts.mutation.auto_import_model as auto_import_model
    # predict_model = auto_import_model.auto_import_msmodel()
    
    ckpt_name = tuple(model_path.split("/"))[-1]
    network_cls = get_cls_through_file(model_path, ckpt_name)
    predict_model = network_cls()
    
    ckpt_path = model_path + "/" + ckpt_name + ".ckpt"
    param_dict = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(predict_model, param_dict)
    model_predict = mindspore.Model(network=predict_model)
    main_logger.info("INFO:load model and compile done!")
    #model prediction
    pred_dataset = pred_dataset.batch(batch_size=batch_num)
    pred_list = []
    for i, d in enumerate(pred_dataset.create_dict_iterator()): 
        test_data = d["image"]
        if i == 0:
            res = model_predict.predict(test_data)
            pred_list.append(res)
            continue
        else:
            res1 = model_predict.predict(test_data)
            pred_list.append(res1)
    concat_op = mindspore.ops.Concat(0)
    res = concat_op(pred_list)
    res_numpy = res.asnumpy()
    # print("===========================")
    # print(res_numpy)
    # print("===========================")
    # import numpy as np
    # np.save(res)
    # main_logger.info("SUCCESSFULLY save res")
    # res = predict_model.predict(test_x,batch_size=batch_size)，
    # test_x表示将要预测的数据集，batch_size表示一次性输入多少张图片给网络进行训练。会返回每个测试集预测各个类别的概率
    main_logger.info("SUCCESS:Get prediction for {} successfully on {}!".format(mut_model_name,bk))
    """Store prediction result to redis"""
    redis_conn.hset("prediction_{}".format(mut_model_name),bk,pickle.dumps(res_numpy))


if __name__ == "__main__":

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str, help="name of backends")
    parse.add_argument("--exp", type=str, help="experiments identifiers")
    parse.add_argument("--test_size", type=int, help="amount of testing image")
    parse.add_argument("--model", type=str, help="path of the model to predict")
    parse.add_argument("--redis_db", type=int)
    parse.add_argument("--config_name", type=str)
    flags, unparsed = parse.parse_known_args(sys.argv[1:])
    """Load Configuration"""
    warnings.filterwarnings("ignore")
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{flags.config_name}")
    pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'],db=flags.redis_db)
    redis_conn = redis.Redis(connection_pool=pool)

    parameters = lemon_cfg['parameters']
    gpu_ids = parameters['gpu_ids']
    gpu_list = parameters['gpu_ids'].split(",")

    """Init cuda"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    warnings.filterwarnings("ignore")

    batch_size= 32
    """Switch backend"""
    bk_list = ['mindspore1.6.2', 'mindspore1.7.1', 'mindspore1.8.1'] #'tensorflow', 'theano', 'cntk','mxnet', 
    bk = flags.backend
    os.environ['KERAS_BACKEND'] = bk
    os.environ['PYTHONHASHSEED'] = '0'
    # if bk == 'tensorflow':
    #     os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
    #     import tensorflow as tf
    #     main_logger.info(tf.__version__)
    #     batch_size = 128
    #     import keras
    # if bk == 'theano':
    #     if len(gpu_list) == 2:
    #         os.environ['THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]};dev{gpu_list[1]}->cuda{gpu_list[1]}," \
    #                                      f"force_device=True,floatX=float32,lib.cnmem=1"
    #     else:
    #         os.environ['THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]}," \
    #                                      f"force_device=True,floatX=float32,lib.cnmem=1"
    #     import theano as th
    #     import keras
    #     main_logger.info(th.__version__)
    # if bk == "cntk":
    #     from cntk.device import try_set_default_device,gpu
    #     try_set_default_device(gpu(int(gpu_list[0])))
    #     import cntk as ck
    #     main_logger.info(ck.__version__)
    #     import keras
    # if bk == "mxnet":
    #     import mxnet as mxnet
    #     main_logger.info(f"mxnet_version {mxnet.__version__}")
    #     import keras
    #     batch_size = 16
    
    #adding branches for mindspore
    if bk == "mindspore1.6.2":
        import mindspore as mp
        main_logger.info(f"mindspore_version {mp.__version__}") #get the version of mindspore
        # import keras
    
    if bk == "mindspore1.7.1":
        import mindspore as mp
        main_logger.info(f"mindspore_version {mp.__version__}") #get the version of mindspore

    if bk == "mindspore1.8.1":
        import mindspore as mp
        main_logger.info(f"mindspore_version {mp.__version__}")
        
    if bk == "mindspore1.8.0":
        import mindspore as mp
        main_logger.info(f"mindspore_version {mp.__version__}")

    # from keras import backend as K
    try:
        """Get model prediction"""
        # main_logger.info("INFO:Using {} as backend for states extraction| {} is wanted".format(K.backend(),bk))
        main_logger.info("Using {} as backend.".format(bk))
        dataset, dataset_name= DataUtils.get_data_by_exp_with_bk(flags.exp, flags.test_size, bk, cfg_name=flags.config_name)
        mut_model_name = os.path.split(flags.model)[-1]
        _get_prediction(bk=bk, pred_dataset = dataset, model_path=flags.model, batch_num=batch_size)
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(-1)
