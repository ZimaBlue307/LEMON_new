import os
# from data_utils import Dataset
# from common_utils import accuracy_analysis, onnx_executor, template, MetricsUtils
from datetime import datetime
import onnx
import numpy as np
import logging
import sys
import getopt
import os




#例如： model_need_import = /home/lemon_proj/lyh/LEMON_new/lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_origin0
def auto_generate_import_model_script(model_need_import):
    entire_model_name = tuple(model_need_import.split("/"))[-1] #resnet20_cifar100_origin0
    model_name1 = tuple(entire_model_name.split("_"))[0] #resnet20, 使用model_name1防止修改外面的变量
    dataset_name1 = tuple(entire_model_name.split("_"))[1] #cifar100, 使用dataset_name1防止修改外面的变量
    tmp_dir = tuple(model_need_import.split("/"))[-4]
    model_before_path = model_need_import + "/" + entire_model_name + ".py" 
    # print("model_need_import: ", model_need_import)
    # /home/lemon_proj/lyh/LEMON_new/lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_origin0/resnet20_cifar100_origin0.py
    # model_after_path = "lemon_outputs/" + "{}_{}/".format(model_name1, dataset_name1) + "mut_model/" + entire_model_name + "/" + entire_model_name + ".py"
    ckpt_before_name = model_need_import + "/" + "{}_{}".format(model_name1, dataset_name1) + "_origin0.ckpt"
    # /home/lemon_proj/lyh/LEMON_new/lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_origin0/resnet20_cifar100_origin0.ckpt
    ckpt_after_name = model_need_import + "/" + entire_model_name + ".ckpt"

    #rename ckpt file
    rename_ckpt = "mv {} {}".format(ckpt_before_name, ckpt_after_name)
    #os.system(rename_ckpt)
    
    # whether_in_lemon_out = "lemon_outputs" in model_need_import
    # if whether_in_lemon_out == False:
    #     cp_model_command = f"cp {model_before_path} {model_after_path}"
    #     cp_ckpt_command = f"cp {ckpt_before_path} {ckpt_after_path}"
    #     os.system(cp_model_command)
    #     os.system(cp_ckpt_command)
    
    output_dir = "lemon_outputs"
    mut_model = "mut_model"

    template =  'import mindspore\n' \
                'import numpy as np\n' \
                'def auto_import_msmodel():\n' \
                '\tauto_model = MindSporeModel()\n' \
                '\t# param_dict = mindspore.load_checkpoint("{}")\n' \
                '\t# mindspore.load_param_into_net(auto_model, param_dict)\n' \
                '\treturn auto_model'.format(ckpt_after_name)
                


    # from lemon_outputs.resnet20_cifar100.mut_model.resnet20_cifar100_origin0.resnet20_cifar100_origin0 import MindSporeModel
    import_template = '#model_name,dataset_name,mut_model,entire_model_name = "{}","{}", "{}", "{}"\n' \
        'import sys\n' \
        'sys.path.append("..")\n' \
        'sys.path.append("..")\n' \
        'from {}.{}.{}_{}.{}.{}.{} import MindSporeModel\n'.format(
        model_name1, dataset_name1,
        mut_model,
        entire_model_name,

        output_dir, tmp_dir,
        model_name1, dataset_name1,
        mut_model,
        entire_model_name,
        entire_model_name)
    script = import_template + template

    script_path = f"./scripts/mutation" #配合model_mutation_generators.py在根目录下执行
    if not os.path.exists(script_path):
        os.makedirs(script_path)
    # with open(f"{script_path}/auto_import_{model_need_import}.py", "w") as fw:
    #     fw.write(script)
    with open(f"{script_path}/auto_import_model.py", "w") as fw:
        fw.write(script)
    
# if __name__ == "__main__":
#     model_need_import = "/home/lemon_proj/lyh/LEMON_new/lemon_outputs/resnet20_cifar100/mut_model/resnet20_cifar100_orig_NAI2"
#     auto_generate_import_model_script(model_need_import)