# 以调用tree_1.py中的函数作为主要的测试和实现方式
import inspect
import os
import ast
import astunparse
import copy
import sys
import astor
import json
import mindspore

def get_model_input_index(input_list, analyzed_data):
    """
    for example, 
    input_list: ['opt_conv2d_51', 'module3_1_opt'] or ['module5_0.module0_0.opt_batchnorm2d_0']
    Each input_list can be obtained from the input element in analyzed_data[i]
    analyzed_data is the same as file analyzed_data.json;
    return a return_list, return_list[i] is the input index of input_list[i]; and len(return_list) equals to len(input_list)
    """
    return_list = list()
    for i, input in enumerate(input_list):
        if 'input' in input:
            # print("This is the input.")
            return_list.append(-1)
        else:
            input_tuple = tuple(input.split("."))
            input_name = input_tuple[-1] #最后一位是输入的name
            input_prefix = '.'.join(input[:-1])#前缀用来筛选，防止出现相同的name
            # print(input_name)
            # print(input_prefix)
            # print("===========")
            for i, element in enumerate(analyzed_data):
                if (input_name != 'x') and (input_name == element[0]) and (input_prefix in element[1]):
                    return_list.append(i)
                    break
                elif (input_name == 'x') and (input_prefix in element[1]): #往上找到第一个的input的index;
                    return_list.append(i)
                    break
                else:    
                    continue
    return return_list

def test_get_model_input_index():
    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)
    for i, element in enumerate(analyzed_data):
        print(i)
        print(element)
        input_list = element[-1]
        return_list = get_model_input_index(input_list, analyzed_data)
        element.append(return_list)
    print("================")
    empty_list = []
    for i, element in enumerate(analyzed_data):
        if element[-1] == []:
            empty_list.append(i)
            print(element)
    # 现在剩a=b+c的格式没保存；
    print(empty_list)

def get_model_output_index(data_element, analyzed_data):
    return_list = []

    output_name = data_element[0]
    op_tuple = tuple(data_element[1].split("."))
    op_prefix = '.'.join(op_tuple[:-1])
    if 'ast' in data_element[1]: #处理一些特殊情况；
        input_search = output_name
    elif len(op_prefix) != 0:
        input_search = op_prefix + "." + output_name
    else:
        input_search = output_name
    #默认analyzed_data的最后一条元素是最终的输出
    if data_element == analyzed_data[-1]:
        return_list.append(-1)
    #先考虑相同class之内的；
    for i, element in enumerate(analyzed_data):
        input_list = element[-1]
        for j, input in enumerate(input_list):
            if input_search == input:
                return_list.append(i)
                break
            else:
                continue
        if len(return_list) != 0:
            break
    #再考虑class跳转出去的：
    if len(return_list) == 0:
        for i, element in enumerate(analyzed_data):
            entire_op_name = element[1]
            if entire_op_name == op_prefix:
                return_list.append(i)
            if len(return_list) != 0:
                break
    # 最后可能还要考虑其他的特殊情况，需要不断补充
    return return_list

def test_get_model_output_index():
    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)
    for i, data_element in enumerate(analyzed_data):
        return_list = get_model_output_index(data_element, analyzed_data)
        data_element.append(return_list)
        print(data_element)
    print("=============")
    empty_list = []
    for i, element in enumerate(analyzed_data):
        if element[-1] == []:
            empty_list.append(i)
            print(element)
    # 现在剩a=b+c的格式没保存；
    print(empty_list)

if __name__ == "__main__":
    test_get_model_output_index()