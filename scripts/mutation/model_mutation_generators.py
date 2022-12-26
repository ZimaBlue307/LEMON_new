import pickle
import sys
from scripts.mutation.model_mutation_operators import *
#from model_mutation_operators import *
import argparse
from scripts.mutation.hyr_import_ms_model import auto_generate_import_model_script
from scripts.tools.utils import *
import os
import warnings
import astor
import ast

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
mylogger = Logger()

def generate_model_by_model_mutation(model, operator,mutate_ratio=0.3):
    """
    Generate models using specific mutate operator
    :param model: model loaded by mindspore ('mindspore1.6.2', mindspore1.7.1 and mindspore1.8.1)
    :param operator: mutation operator
    :param mutate_ratio: ratio of selected neurons
    :return: mutation model object
    """
    if operator == 'WS':
        mutate_indices = utils.ModelUtils.weighted_layer_indices(model)
        mylogger.info("Generating model using {}".format(operator))
        return WS_mut(model=model,mutation_ratio=mutate_ratio,mutated_layer_indices=mutate_indices)
    elif operator == 'GF':
        mylogger.info("Generating model using {}".format(operator))
        return GF_mut(model=model,mutation_ratio=mutate_ratio)
    elif operator == 'NEB':
        mylogger.info("Generating model using {}".format(operator))
        return NEB_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NAI':
        mylogger.info("Generating model using {}".format(operator))
        return NAI_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NS':
        mylogger.info("Generating model using {}".format(operator))
        return NS_mut(model=model)
    elif operator == 'ARem':
        mylogger.info("Generating model using {}".format(operator))
        return ARem_mut(model=model)
    elif operator == 'ARep':
        mylogger.info("Generating model using {}".format(operator))
        return ARep_mut(model=model)

    elif operator == 'None':
        print("just for test")
    else:
        mylogger.info("No such Mutation operator {}".format(operator))
        return None


def generate_model_by_inter_mutation(model_path, operator):
    """
    Generate models using specific mutate operator
    :param model_path: model_path saved in output file ('mindspore1.6.2', mindspore1.7.1 and mindspore1.8.1)
    :param operator: mutation operator
    :param mutate_ratio: ratio of selected neurons
    :return: mutation model object
    """
    if operator == 'LC':
        mylogger.info("Generating model using {}".format(operator))
        return LC_mut(model_path=model)
    elif operator == 'LA':
        mylogger.info("Generating model using {}".format(operator))
        return LA_mut(model_path=model)
    elif operator == 'LR':
        mylogger.info("Generating model using {}".format(operator))
        return LR_mut(model_path=model)
    elif operator == 'LS':
        mylogger.info("Generating model using {}".format(operator))
        return LS_mut(model_path=model)
    elif operator == 'MLA':
        mylogger.info("Generating model using {}".format(operator))
        return MLA_mut(model_path=model)
    elif operator == 'None':
        print("just for test")
    else:
        mylogger.info("No such Mutation operator {}".format(operator))
        return None

def all_mutate_ops():
    return ['WS','GF','NEB','NAI','NS','ARem','ARep','LA','LC','LR','LS','MLA']

def save_IRtable(model_path, table):
    # get model_ast
    model_ast = astor.parse_file(model_path)
    # get analyzed_data
    analyzed_data = None
    # get module_dict
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item
    # construct our table
    # table = utils.construct_table(model_ast, analyzed_data, module_dict)
    table_save_tuple = tuple(model_path.split("/"))
    table_name = table_save_tuple[-1]
    table_name = tuple(table_name.split("."))[0]
    table_save_path = '/'.join(table_save_tuple[:-1]) + "/" + table_name + "_table.pkl"
    # print(table_save_path)
    with open(table_save_path, 'wb') as file1:
        pickle.dump(table, file1)
    # with open(table_save_path, 'rb') as file2:
    #     new_table = pickle.load(file2)
    # return new_table
    

if __name__ == '__main__':

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path", type=str, help="model path") #文件夹名字
    parse.add_argument("--mutate_op", type=str, help="model mutation operator")
    # add argument: checkpoint path，文件夹名字
    #reference website: 
    #https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.load_param_into_net.html?highlight=load_checkpoint
    parse.add_argument("--checkpoint_path", type=str, help="model checkpoint path")
    parse.add_argument("--save_path", type=str, help="model save path")
    parse.add_argument("--mutate_ratio", type=float, help="mutate ratio")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    #import keras
    import mindspore
    model_path = flags.model_path
    checkpoint_path = flags.checkpoint_path
    mutate_ratio = flags.mutate_ratio
    # add argument checkpoint path
    print("Current {}; Mutate ratio {}".format(flags.mutate_op,mutate_ratio))
    # model_path: data/lemon_outputs/alexnet-cifar10/mut_model/alexnet_cifar10_LA1
    model_name = tuple(model_path.split("/"))[-1] #例如resnet20_cifar100
    ckpt_name = model_name + '.ckpt'
    ckpt_name = os.path.join(model_path, ckpt_name)

    # get ir table
    table_path = os.path.join(model_path, "analyzed_data.json")
    with open(table_path, 'rb') as f:
        table = json.load(f)
    # ckpt_name = tuple(checkpoint_path.split("/"))[-1] 
    # checkpoint =checkpoint_path + "/" + ckpt_name + ".ckpt"
    # param_dict = mindspore.load_checkpoint(checkpoint)
    auto_generate_import_model_script(model_path)
    from scripts.mutation.auto_import_model import *
    origin_model = auto_import_msmodel()
    param_dict = mindspore.load_checkpoint(ckpt_name)
    mindspore.load_param_into_net(origin_model, param_dict)
    if flags.mutate_op in {'GF', 'WS', 'NEB', 'NAI', 'NS'}:
        mutated_model = generate_model_by_model_mutation(model=origin_model, operator=flags.mutate_op, mutate_ratio=mutate_ratio)
        if mutated_model is None:
            raise Exception("Error: Model mutation using {} failed".format(flags.mutate_op))
        else:
            # 先考虑层内算子
            mylogger.info('seems into this branch {}'.format(flags.mutate_op))
            save_path = flags.save_path
            ckpt_name = tuple(save_path.split("/"))[-1]
            ckpt_save_path = save_path + "/" + ckpt_name + ".ckpt"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            mindspore.save_checkpoint(mutated_model, ckpt_save_path)  # 保存ckpt文件
            # print("ckpt_save_path: ", ckpt_save_path)
            # 保存py文件
            old_model_path = model_path + "/" + model_name + ".py"
            new_model_name = tuple(save_path.split("/"))[-1]
            new_model_path = save_path + "/" + new_model_name + ".py"
            copy_command = f"cp {old_model_path} {new_model_path}"
            os.system(copy_command)
            # print("===============now saving IRTable===============")
            # print("new_model_path: ", new_model_path)
            # print("table: ", table)
            mylogger.info("-----saving table-----")
            save_IRtable(new_model_path, table)
    else:
        # 再考虑层间变异算子
        mutated_ast = generate_model_by_inter_mutation(model_path=origin_model, operator=flags.mutate_op)
        mylogger.info("using inter-layer ops, saving!")
        new_model_name = tuple(flags.save_path.split("/"))[-1]
        new_model_path = flags.save_path + "/" + new_model_name + ".py"
        
        print("===============now saving IRTable===============")
        print("new_model_path: ", new_model_path)
        print("table: ", table)
        save_IRtable(new_model_path, table)