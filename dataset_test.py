import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mindspore.dataset.transforms as transforms
import mindspore
import mindspore.numpy as msnp
from lemon_outputs.resnet20_cifar100.mut_model.resnet20_cifar100_origin0.resnet20_cifar100_origin0 import MindSporeModel


def get_data_by_exp(exp, test_size):
    if 'cifar100' in exp:
        cifar100_dir = "dataset/cifar100/cifar-100-binary"
        dataset = ds.Cifar100Dataset(dataset_dir = cifar100_dir, usage='test', num_samples = test_size, shuffle=False) # batch_size=32, download=True
        import mindspore.dataset.transforms as C
        import mindspore.dataset.vision as CV
        from mindspore.dataset.vision import Inter
        resize_height, resize_width = 32, 32
        rescale_param = 1.0 / 255.0
        shift_param = -1.0
        one_hot_opt = C.OneHot(num_classes=100) 
        rescale_op = CV.Rescale(rescale_param, shift_param)
        resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
        dataset = dataset.map(operations = one_hot_opt, input_columns=["fine_label"]) #把细标签转换为独热编码
        dataset = dataset.map(operations = rescale_op, input_columns=["image"])
        dataset = dataset.map(operations = resize_op, input_columns=["image"])
    return dataset

def concat_dataset(dataset, dataset_name, test_size):
    import mindspore
    if dataset_name == "cifar100":
        for i, data in enumerate(dataset.create_dict_iterator()):
            label_tensor = data['fine_label']
            # print(np.shape(label_tensor))
            # label_tensor = mindspore.numpy.expand_dims(label_tensor, 0)
            break
        for i, data in enumerate(dataset.create_dict_iterator()):
            if i == 0: 
                continue
            data = data['fine_label']
            label_tensor = mindspore.ops.concat((label_tensor, data))
            if i == test_size-1:
                break
        return label_tensor

def delta(y1_pred, label_tensor, y_true=None):
        import mindspore
        y1_pred = np.reshape(y1_pred, [np.shape(y1_pred)[0], -1])
        #np.reshape will change the datatype to object.
        label_tensor = mindspore.Tensor(label_tensor, dtype=mindspore.float32)
        label_tensor = np.reshape(label_tensor, [np.shape(label_tensor)[0], -1])
        # now the datatypes of y1_pred and label_tensor are all object
        mean_ans = np.mean(np.abs(y1_pred - label_tensor), axis = 1)
        sum_ans = np.sum(np.abs(y1_pred - label_tensor), axis=1)
        return mean_ans, sum_ans


if __name__ == "__main__":
    dataset_name = "cifar100"
    test_size = 16
    batch_num = 2
    dataset = get_data_by_exp(dataset_name, test_size)
    dataset = dataset.batch(batch_size=batch_num)
    
    ckpt_path = "lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_origin0/resnet20_cifar100_origin0.ckpt"
    resnet20_cifar100 = MindSporeModel()
    param_dict = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(resnet20_cifar100, param_dict)
    model_predict = mindspore.Model(network=resnet20_cifar100)
    
    count = 0
    for d in dataset.create_dict_iterator(): 
        count+=1
        test_data = d["image"]
        if count==1:
            res1 = model_predict.predict(test_data)
            # res1 = mindspore.numpy.expand_dims(res1, 0)
            # print(np.shape(res1))
            continue
        else:
            res = model_predict.predict(test_data)
            res1 = mindspore.ops.concat((res1, res))
            
    print(np.shape(res1))
    label_tensor = concat_dataset(dataset, dataset_name, test_size)
    print(np.shape(label_tensor))
    
    mean_ans, sum_ans = delta(res1, label_tensor)