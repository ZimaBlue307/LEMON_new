# -*-coding:UTF-8-*-
from scripts.logger.lemon_logger import Logger
from scripts.tools.mutator_selection_logic import Roulette, MCMC
from scripts.tools.utils import Table
import argparse
import sys
import ast
import os
import numpy as np
from itertools import combinations
import redis
import pickle
from scripts.tools import utils
import shutil
import re
import datetime
import configparser
import warnings
import math
from scripts.mutation.model_shape_utils import ShapeUitls
import json
import astor

np.random.seed(20200501)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def partially_nan_or_inf(predictions, bk_num):
    """
    Check if there is NAN in the result
    """
    def get_nan_num(nds):
        _nan_num = 0
        for nd in nds:
            if np.isnan(nd).any() or np.isinf(nd).any():
                _nan_num += 1
        return _nan_num

    if len(predictions) == bk_num:
        for input_predict in zip(*predictions):
            nan_num = get_nan_num(input_predict)
            if 0 < nan_num < bk_num:
                return True
            else:
                continue
        return False
    else:
        raise Exception("wrong backend amounts")


def get_selector_by_startegy_name(mutator_s, mutant_s):

    mutant_strategy_dict = {"ROULETTE":Roulette}
    mutator_strategy_dict = {"MCMC":MCMC}
    return mutator_strategy_dict[mutator_s],mutant_strategy_dict[mutant_s]


def save_mutate_history(selector, invalid_history: dict, mutant_history: list):
    mutator_history_path = os.path.join(experiment_dir, "mutator_history.csv")
    mutant_history_path = os.path.join(experiment_dir, "mutant_history.txt")
    with open(mutator_history_path, "w+") as fw:
        fw.write("Name,Success,Invalid,Total\n")
        for op in invalid_history.keys():
            mtrs = selector.mutators[op]
            invalid_cnt = invalid_history[op]
            fw.write("{},{},{},{}\n".format(op, mtrs.delta_bigger_than_zero, invalid_cnt, mtrs.total))
    with open(mutant_history_path, "w+") as fw:
        for mutant in mutant_history:
            fw.write("{}\n".format(mutant))


def is_nan_or_inf(t):
    if math.isnan(t) or math.isinf(t):
        return True
    else:
        return False

def continue_checker(**run_stat):
    start_time = run_stat['start_time']
    time_limitation = run_stat['time_limit']
    cur_counters = run_stat['cur_counters']
    counters_limit = run_stat['counters_limit']
    s_mode = run_stat['stop_mode']

    # if timing
    if s_mode == 'TIMING':
        hours, minutes, seconds = utils.ToolUtils.get_HH_mm_ss(datetime.datetime.now() - start_time)
        total_minutes = hours * 60 + minutes
        mutate_logger.info(f"INFO: Mutation progress: {total_minutes}/{time_limitation} Minutes!")
        if total_minutes < time_limitation:
            return True
        else:
            return False
    # if counters
    elif s_mode == 'COUNTER':
        if cur_counters < counters_limit:
            mutate_logger.info("INFO: Mutation progress {}/{}".format(cur_counters + 1, counters_limit))
            return True
        else:
            return False
    else:
        raise Exception(f"Error! Stop Mode {s_mode} not Found!")


def _generate_and_predict(res_dict, filename, mutate_num, mutate_ops, test_size, exp, backends, cfg_name):
    """
    Generate models using mutate operators and store them
    """
    mutate_op_history = {k: 0 for k in mutate_ops}
    mutate_op_invalid_history = {k: 0 for k in mutate_ops}
    mutant_history = []

    # get mutator selection strategy
    # origin_model_name = "{}_origin0".format(exp)
    origin_model_name = "{}_origin".format(exp)
    origin_save_path = os.path.join(mut_dir, origin_model_name)
    mutator_selector_func, mutant_selector_func = get_selector_by_startegy_name(mutator_strategy,mutant_strategy)
    # [origin_model_name] means seed pool only contains initial model at beginning.
    mutator_selector, mutant_selector = mutator_selector_func(mutate_ops), mutant_selector_func([origin_model_name],
                                                                                                capacity=mutate_num + 1)
    if not os.path.exists(origin_save_path):
        # copy model file and analyze shape info
        mutate_logger.info(f"start copy origin model from {filename} to {origin_save_path}")
        shutil.copytree(filename, origin_save_path)
        #model_analyze可以py文件中插装得到shape信息的，得到json文件
        table_path = ShapeUitls.model_analyze(origin_save_path, "mindspore-1.8.1", cfg_name)
        with open(table_path, 'rb') as f:
            analyzed_data = json.load(f)
        model_prefix = origin_save_path.split("/")[-1]
        model_path = model_prefix + '.py'
        model_path = os.path.join(origin_save_path, model_path)
        network_ast = astor.code_to_ast.parse_file(model_path)
        # construct table
        module_dict = Table.construct_module_dict(network_ast)
        table = Table.construct_table(network_ast, analyzed_data, module_dict)
        table_save_tuple = tuple(model_path.split("/"))
        table_name = table_save_tuple[-1]
        table_name = tuple(table_name.split("."))[0]
        table_save_path = '/'.join(table_save_tuple[:-1]) + "/" + table_name + '_table.pkl'
        with open(table_save_path, "wb") as file1:
            pickle.dump(table, file1)

    # print("===========================")
    # print("generate the json of model {}".format(origin_model_name))
    # print("===========================")
    
    origin_model_status, res_dict, accumulative_inconsistency, _ = get_model_prediction(res_dict,
                                                                                        origin_save_path,
                                                                                        origin_model_name, exp,
                                                                                        test_size, backends)
    if not origin_model_status:
        mutate_logger.error(f"Origin model {exp} crashed on some backends! LEMON would skip it")
        sys.exit(-1)

    last_used_mutator = None
    last_inconsistency = accumulative_inconsistency
    mutant_counter = 0

    start_time = datetime.datetime.now()
    order_inconsistency_dict = {}
    run_stat = {'start_time': start_time, 'time_limit': time_limit, 'cur_counters': mutant_counter,
                'counters_limit': mutate_num, 'stop_mode': stop_mode}

    while continue_checker(**run_stat):
        picked_seed = utils.ToolUtils.select_mutant(mutant_selector)
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator)
        mutate_op_history[selected_op] += 1
        last_used_mutator = selected_op
        mutator = mutator_selector.mutators[selected_op]
        mutant = mutant_selector.mutants[picked_seed]

        #new seed name is a folder name, including ckpt and python file
        new_seed_name = "{}_{}{}".format(picked_seed[:-3], selected_op, mutate_op_history[selected_op])
        # seed name would not be duplicate
        if new_seed_name not in mutant_selector.mutants.keys():
            new_seed_path = os.path.join(mut_dir, new_seed_name)
            picked_seed_path = os.path.join(mut_dir, picked_seed)
            mutate_st = datetime.datetime.now()
            mutate_status = os.system("{}/lemon/bin/python -m  scripts.mutation.model_mutation_generators --model_path {} "
                                      "--mutate_op {} --checkpoint_path {} --save_path {} --mutate_ratio {}".format(python_prefix,
                                                                                               picked_seed_path,
                                                                                               selected_op,
                                                                                               picked_seed_path,
                                                                                               new_seed_path,
                                                                                               flags.mutate_ratio))
            mutate_et = datetime.datetime.now()
            mutate_dt = mutate_et - mutate_st
            h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
            mutate_logger.info("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))
            # mutation status code is successful
            if mutate_status == 0:
                mutant.selected += 1
                mutator.total += 1
                
                # execute this model on all platforms
                predict_status, res_dict, accumulative_inconsistency, model_outputs = \
                    get_model_prediction(res_dict, new_seed_path, new_seed_name, exp, test_size, backends)

                if predict_status:
                    mutant_history.append(new_seed_name)

                    if utils.ModelUtils.is_valid_model(inputs_backends=model_outputs,backends_nums=len(backends)):
                        delta = accumulative_inconsistency - last_inconsistency

                        if mutator_strategy == 'MCMC':
                            mutator.delta_bigger_than_zero = mutator.delta_bigger_than_zero + 1 \
                                if delta > 0 else mutator.delta_bigger_than_zero

                        if mutant_strategy == 'ROULETTE' and delta > 0:
                            # when size >= capacity:
                            # random_mutant & Roulette would drop one and add new one
                            if mutant_selector.is_full():
                                mutant_selector.pop_one_mutant()
                            mutant_selector.add_mutant(new_seed_name)
                            last_inconsistency = accumulative_inconsistency

                        mutate_logger.info("SUCCESS:{} pass testing!".format(new_seed_name))
                        mutant_counter += 1
                    else:
                        mutate_op_invalid_history[selected_op] += 1
                        mutate_logger.error("Invalid model Found!")
                else:
                    mutate_logger.error("Crashed or NaN model Found!")
            else:
                mutate_logger.error("Exception raised when mutate {} with {}".format(picked_seed, selected_op))

            mutate_logger.info("Mutated op used history:")
            mutate_logger.info(mutate_op_history)

            mutate_logger.info("Invalid mutant generated history:")
            mutate_logger.info(mutate_op_invalid_history)

        run_stat['cur_counters'] = mutant_counter

    save_mutate_history(mutator_selector, mutate_op_invalid_history, mutant_history)

    return res_dict


#res_dict: 例如{'D_MAD': {}} <class 'dict'>
#model_idntfr: 例如resnet20_cifar100_orig <class 'str'>
def generate_metrics_result(res_dict, predict_output, model_idntfr, test_size):
    mutate_logger.info("Generating Metrics Result")
    accumulative_incons = 0
    backends_pairs_num = 0
    dataset_name = model_idntfr.split("_")[1]
    # Compare results pair by pair
    for pair in combinations(predict_output.items(), 2):
        backends_pairs_num += 1
        backend1, backend2 = pair
        bk_name1, prediction1 = backend1
        bk_name2, prediction2 = backend2
        bk_pair = "{}_{}".format(bk_name1, bk_name2)#例如mindspore1.7.1_mindspore1.8.1
        for metrics_name, metrics_result_dict in res_dict.items():
            metrics_func = utils.MetricsUtils.get_metrics_by_name(metrics_name)
            # metrics_results in list type
            # metrics_results = metrics_func(prediction1, prediction2, y_test[:flags.test_size])
            # print(type(prediction1))
            # print(prediction2)
            metrics_results = metrics_func(prediction1, prediction2, dataset, dataset_name, test_size)
            
            mutate_logger.info(f"get metrics_results from backend pair: {bk_pair}")
            # ACC -> float: The sum of all inputs under all backends
            # print(type(metrics_results)) <class 'list'>
            # print(metrics_results)
            accumulative_incons += sum(metrics_results)
            for input_idx, delta in enumerate(metrics_results):
                delta_key = "{}_{}_{}_input{}".format(model_idntfr, bk_name1, bk_name2, input_idx)
                metrics_result_dict[delta_key] = delta

    mutate_logger.info(f"Accumulative Inconsistency: {accumulative_incons}")
    return res_dict, accumulative_incons



# origin_model_status, res_dict, accumulative_inconsistency, _ = get_model_prediction(res_dict,
#                                                                                         origin_save_path,
#                                                                                         origin_model_name, exp,
#                                                                                         test_size, backends)
def get_model_prediction(res_dict, model_path, model_name, exp, test_size, backends):
    """
    Get model prediction on different backends and calculate distance by metrics
    """
    predict_output = {b: [] for b in backends}
    model_idntfr = model_name[:-3]
    all_backends_predict_status = True
    for bk in backends:
        python_bin = f"{python_prefix}/{bk}/bin/python"
        predict_st = datetime.datetime.now()
        pre_status_bk = os.system(f"{python_bin} -u -m run.patch_prediction_extractor --backend {bk} "
                                  f"--exp {exp} --test_size {test_size} --model {model_path} "
                                  f"--redis_db {lemon_cfg['redis'].getint('redis_db')} --config_name {flags.config_name}")
        predict_et = datetime.datetime.now()
        predict_td = predict_et - predict_st
        h, m, s = utils.ToolUtils.get_HH_mm_ss(predict_td)
        mutate_logger.info("Prediction Time Used on {} : {}h, {}m, {}s".format(bk, h, m, s))

        # If no exception is thrown,save prediction result
        if pre_status_bk == 0:
            data = pickle.loads(redis_conn.hget("prediction_{}".format(model_name), bk))
            predict_output[bk] = data
            # why this data variable can not be printed?
        # record the crashed backend
        else:
            all_backends_predict_status = False
            mutate_logger.error("{} crash on backend {} when predicting ".format(model_name, bk))
    status = False
    accumulative_incons = None

    # run ok on all platforms
    if all_backends_predict_status:
        predictions = list(predict_output.values()) # predictions长度为2；
        # print(len(predictions))
        # 为什么这个prediction打印不出来呢？
        # model_idntfr = resnet20_cifar100_orig
        res_dict, accumulative_incons = generate_metrics_result(res_dict=res_dict, predict_output=predict_output, model_idntfr=model_idntfr, test_size=test_size)

        # If all backends are working fine, check if there exists NAN or INF in the result
        # `accumulative_incons` is nan or inf --> NaN or INF in results
        if is_nan_or_inf(accumulative_incons):
            # has NaN on partial backends
            if partially_nan_or_inf(predictions, len(backends)):
                nan_model_path = os.path.join(nan_dir, f"{model_idntfr}_NaN_bug.h5")
                mutate_logger.error("Error: Found one NaN bug. move NAN model")

            # has NaN on all backends --> not a NaN bug
            else:
                nan_model_path = os.path.join(nan_dir, f"{model_idntfr}_NaN_on_all_backends.h5")
                mutate_logger.error("Error: Found one NaN Model on all libraries. move NAN model")
            shutil.move(model_path, nan_model_path)

        else:  # No NaN or INF on any backend
            mutate_logger.info("Saving prediction")
            with open("{}/prediction_{}.pkl".format(inner_output_dir, model_idntfr), "wb+") as f:
                pickle.dump(predict_output, file=f)
            status = True

    # save crashed model
    else:
        mutate_logger.error("Error: move crash model")
        crash_model_path = os.path.join(crash_dir, model_name)
        shutil.move(model_path, crash_model_path)

    return status, res_dict, accumulative_incons, predict_output


if __name__ == "__main__":

    starttime = datetime.datetime.now()
    """
    Parser of command args.
    It could make mutate_lemon.py run independently without relying on mutation_executor.py
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--is_mutate", type=ast.literal_eval, default=False,
                       help="parameter to determine mutation option")
    parse.add_argument("--mutate_op", type=str, nargs='+',
                       choices=['WS', 'GF', 'NEB', 'NAI', 'NS', 'ARem', 'ARep', 'LA', 'LC', 'LR', 'LS', 'MLA']
                       , help="parameter to determine mutation option")
    parse.add_argument("--model", type=str, help="relative path of model file(from root dir)")
    parse.add_argument("--output_dir", type=str, help="relative path of output dir(from root dir)")
    parse.add_argument("--backends", type=str, nargs='+', help="list of backends")
    parse.add_argument("--mutate_num", type=int, help="number of variant models generated by each mutation operator")
    parse.add_argument("--mutate_ratio", type=float, help="ratio of mutation")
    parse.add_argument("--exp", type=str, help="experiments identifiers")
    parse.add_argument("--test_size", type=int, help="amount of testing image")
    parse.add_argument("--config_name", type=str, help="config name")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    warnings.filterwarnings("ignore")
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{flags.config_name}")
    time_limit = lemon_cfg['parameters'].getint("time_limit") #默认是60min
    mutator_strategy = lemon_cfg['parameters'].get("mutator_strategy").upper() # MCMC
    mutant_strategy = lemon_cfg['parameters'].get("mutant_strategy").upper() # ROULETTE
    stop_mode = lemon_cfg['parameters'].get("stop_mode").upper() #TIMING
    alpha = lemon_cfg['parameters'].getfloat("alpha")

    mutate_logger = Logger()
    pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'],
                                db=lemon_cfg['redis'].getint('redis_db'))
    redis_conn = redis.Redis(connection_pool=pool)

    for k in redis_conn.keys():
        if flags.exp in k.decode("utf-8"):
            redis_conn.delete(k)

    # exp : like alexnet_cifar100
    experiment_dir = os.path.join(flags.output_dir, flags.exp) #like /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100
    mut_dir = os.path.join(experiment_dir, "mut_model")# /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100/mut_model
    crash_dir = os.path.join(experiment_dir, "crash")# /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100/crash
    nan_dir = os.path.join(experiment_dir, "nan")# /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100/nan
    inner_output_dir = os.path.join(experiment_dir, "inner_output")# /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100/inner_output
    metrics_result_dir = os.path.join(experiment_dir, "metrics_result")# /home/lemon_proj/lyh/LEMON_new/lemon_outputs/alexnet_cifar100/metrics_result
    
    dataset, dataset_name = utils.DataUtils.get_data_by_exp_with_bk(flags.exp, flags.test_size, backend_name = "mindspore1.8.1", cfg_name=flags.config_name)
    mutate_logger.info(f"get test data with {dataset_name}")
    pool_size = lemon_cfg['parameters'].getint('pool_size')
    python_prefix = lemon_cfg['parameters']['python_prefix'].rstrip("/")

    try:
        metrics_list = lemon_cfg['parameters']['metrics'].split(" ") #D_MAD
        lemon_results = {k: dict() for k in metrics_list}
        lemon_results = _generate_and_predict(lemon_results, flags.model, flags.mutate_num, flags.mutate_op,
                                              flags.test_size, flags.exp, flags.backends, flags.config_name)
        with open("{}/{}_lemon_results.pkl".format(experiment_dir, flags.exp), "wb+") as f:
            pickle.dump(lemon_results, file=f)
        utils.MetricsUtils.generate_result_by_metrics(metrics_list, lemon_results, metrics_result_dir, flags.exp)

    except Exception as e:
        mutate_logger.exception(sys.exc_info())

    # from keras import backend as K
    # K.clear_session()

    endtime = datetime.datetime.now()
    time_delta = endtime - starttime
    h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    mutate_logger.info("Mutation process is done: Time used: {} hour,{} min,{} sec".format(h, m, s))
