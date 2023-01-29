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
import astor
from tree_1 import *

class PrintVisitor(ast.NodeVisitor):

    def __init__(self):
        super(PrintVisitor, self).__init__()

    def visit_Assign(self, node):
        '''
        for every assign statement, add a print node after it
        print the following thing
        1. if it is not a module, print the name (e.g. conv1d) and the shape
        2. if it is a module, print the module name
        :param node:
        :return:
        '''

        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if 'construct' in node.name:
            # add a print node after every assign node
            global_str = 'global save'
            global_node = ast.parse(global_str).body[0]
            node.body.insert(0, global_node)


            i = 0
            flag = True
            while flag:
                sub_node = node.body[i]
                if isinstance(sub_node, ast.Assign):
                    if isinstance(sub_node.value, ast.Call): # check the assign node is using a operator, not just a parameter
                        target = sub_node.targets[0].id
                        print_str = None
                        if 'module' in target:
                            # if it is a module, change the statement with a construct statement
                            # print_str = "print('{} is a module')\n".format(target)
                            node_str = astunparse.unparse(sub_node)
                            # print(node_str)
                            node_strs = node_str.split('(')
                            new_str = ''
                            for j in range(len(node_strs)):
                                if j == 0:
                                    new_str = node_strs[j] + '.construct'
                                else:
                                    new_str += '('
                                    new_str += node_strs[j]
                            print(new_str)
                            new_node = ast.parse(new_str).body[0]
                            node.body[i] = new_node
                            module_name = sub_node.value.func.attr
                            args = sub_node.value.args
                            arg_list = list()
                            for arg in args:
                                if isinstance(arg, ast.Name):
                                    arg_list.append(arg.id)
                            pre_stat = "save.append(['{}', '{} start', {}])".format(module_name, module_name, arg_list)
                            aft_stat = "save.append(['{}', '{} end', {}, '{}'])".format(module_name, module_name, arg_list, target)
                            print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target, sub_node.value.func.attr, target, arg_list)
                            pre_node = ast.parse(pre_stat).body[0]
                            aft_node = ast.parse(aft_stat).body[0]
                            print_node = ast.parse(print_str).body[0]
                            node.body.insert(i, pre_node)
                            node.body.insert(i+2, aft_node)
                            node.body.insert(i+2, print_node)
                            # print(astunparse.unparse(node))
                            i = i+2
                        else:
                            # get shape
                            if isinstance(sub_node.value, ast.Call):
                                if isinstance(sub_node.value.func, ast.Attribute):
                                    args = sub_node.value.args
                                    arg_list = list()
                                    for arg in args:
                                        if isinstance(arg, ast.Name):
                                            arg_list.append(arg.id)
                                        elif isinstance(arg, ast.Attribute):
                                            arg_list.append(arg.attr)
                                    print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target, sub_node.value.func.attr, target, arg_list)
                                elif isinstance(sub_node.value.func, ast.Call):
                                    args = sub_node.value.args
                                    arg_list = list()
                                    for arg in args:
                                        if isinstance(arg, ast.Name):
                                            arg_list.append(arg.id)
                                        elif isinstance(arg, ast.Attribute):
                                            arg_list.append(arg.attr)
                                    print_str = "save.append(['{}', '{}', {}.shape, {}])".format(target,
                                                                                             sub_node.value.func.func.attr,
                                                                                             target, arg_list)
                                # sub_node_str = astunparse.unparse(sub_node)
                                # print_str = sub_node_str  + print_str + '\n'
                                print_ast = ast.parse(print_str)
                                print_node = print_ast.body[0]
                                print_ast_str = ast.dump(print_node)
                                node.body.insert(i + 1, print_node)
                                # i += 1
                                # print(print_ast_str)
                            else:
                                print("this line is not a attribute node!")

                    elif isinstance(sub_node.value, ast.BinOp):
                        print("++++++++++Get in binop branch!++++++++++")
                        # if the node is binOp, we need add both left name and right name
                        # into the input list
                        left = sub_node.value.left
                        right = sub_node.value.right
                        target = sub_node.targets[0].id

                        def get_ast_name(node):
                            if isinstance(node, ast.Name):
                                return node.id
                            elif isinstance(node, ast.Attribute):
                                return node.attr

                        left_name, right_name = get_ast_name(left), get_ast_name(right)
                        arg_list = [left_name, right_name]
                        print_str = "save.append(['{}', \"{}\", {}.shape, {}])".format(target,
                                                                                     str(type(sub_node.value.op)),
                                                                                     target, arg_list)
                        print_ast = ast.parse(print_str)
                        print_node = print_ast.body[0]
                        print_ast_str = ast.dump(print_node)
                        node.body.insert(i + 1, print_node)


                elif isinstance(sub_node, ast.Return):
                    flag = False
                #     print_str = "print('end')\n"
                #     print_ast = ast.parse(print_str)
                #     print_node = print_ast.body[0]
                #     print_ast_str = ast.dump(print_node)
                #     node.body.insert(i, print_node)


                i = i+1
        ast.fix_missing_locations(node)
        self.generic_visit(node)
        return node

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

def get_model_output_index(analyzed_data):
    for i, data_element in enumerate(analyzed_data):
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
        #默认analyzed_data的最后一条元素是最终的输出；
        if data_element == analyzed_data[-1]:
            return_list.append(-2) #表示最终输出张量；
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
        data_element.append(return_list)

    #再考虑class跳转出去的，以及其他特殊情况.初步打算倒着遍历；
    length = len(analyzed_data)
    for i in range(length):
        data_element = analyzed_data[length - i - 1]
        return_list = []
        if len(data_element[-1]) == 0: 
            op_tuple = tuple(data_element[1].split("."))
            op_prefix = '.'.join(op_tuple[:-1])
            for j in range(length):
                element = analyzed_data[length - j - 1]
                entire_op_name = element[1]
                if (entire_op_name == op_prefix) and (element[-1] != []):
                    return_list = element[-1]
                    break
            data_element[-1] = return_list
    # 最后可能还要考虑其他的特殊情况，需要不断补充
    return analyzed_data

def test_get_model_output_index():
    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)
    new_analyzed_data = get_model_output_index(analyzed_data)
    for i, data_element in enumerate(new_analyzed_data):
        # return_list = get_model_output_index(data_element, analyzed_data)
        # data_element.append(return_list)
        print(data_element)
    print("=============")
    empty_list = []
    for i, element in enumerate(analyzed_data):
        if element[-1] == []:
            empty_list.append(i)
            print(element)
            
    print(empty_list)

def copy_class(model_ast, ir_table, legal_index_list):
    new_ast = model_ast
    new_ir_table = ir_table # new_ir_table is the table need to change and return;
    mut_index = legal_index_list[0] # get the cell need to mutate;
    target_node = new_ir_table.nodeList[mut_index]
    copy_class_list = target_node.node_module
    # copy class
    for i, class_name in enumerate(copy_class_list):
        for j, node in enumerate(new_ast.body):
            if isinstance(node, ast.ClassDef) and node.name == class_name and node.name != "MindSporeModel":
                copy_node = copy.deepcopy(node)
                copy_node.name = class_name + "_copy"
                new_ast.body.insert(j, copy_node)
    # change the info in new_ir_table
    unique_name = target_node.unique_name
    unique_name_list = tuple(unique_name.split('.'))
    unique_name_prefix = '.'.join(unique_name_list[:-1])
    for node in new_ir_table.nodeList:
        if unique_name_prefix in node.unique_name:
            #修改该节点的信息
            #修改unique name
            new_unique_name = node.unique_name
            new_unique_name_list = tuple(new_unique_name.split('.'))
            for name in new_unique_name_list:
                name = name + "_copy"
            node.unique_name = ".".join(new_unique_name_list)  
            # 修改operator_name
            op_name = node.operator_name
            node.operator_name = op_name + "_copy"
            # 修改node_module
            new_node_module_list = node.node_module
            for module_name in new_node_module_list:
                if module_name == "MindSporeModel":
                    continue
                else:
                    module_name = module_name + "_copy"
            node.node_module = new_node_module_list
            # 修改output_name
            out_name = node.output_name
            node.output_name = out_name + "_copy"
    return new_ast, new_ir_table

def test_copy_class():
    # get model_ast
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    model_ast = astor.parse_file(model_path)
    # get analyzed_data
    with open('analyzed_data.json', 'r') as f:
        analyzed_data = json.load(f)
    # get module_dict
    module_dict = dict()
    for item in model_ast.body:
        if isinstance(item, ast.ClassDef):
            module_dict[item.name] = item
    ir_table = construct_table(model_ast, analyzed_data, module_dict)
    legal_index_list = [] # shuffle之后的index；
    new_ast = copy_class(model_ast, ir_table, legal_index_list)
    print(new_ast)

def origin_MLA_mut(model, new_layers = None, mutated_layer_indices=None):
    # mutiple layers addition
    layer_matching = LayerMatching()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    MLA_model = utils.ModelUtils.model_copy(model, 'MLA')
    insertion_points = _MLA_model_scan(model, new_layers, mutated_layer_indices)
    mylogger.info(insertion_points)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                             .format(insertion_points[key], key, type(model.layers[key])))

    # use logic: randomly select a new layer available to insert into the layer which can be inserted
    layers_index_avaliable = list(insertion_points.keys())
    # layer_index_to_insert = np.max([i for i in insertion_points.keys()])
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('choose to insert {} after {}'.format(layer_name_to_insert, MLA_model.layers[layer_index_to_insert].name))
    # insert new layers
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(MLA_model.layers):
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += "_copy"
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output.shape.as_list()
                layers_to_insert = layer_matching.layer_concats[layer_name_to_insert](output_shape)
                for layer_to_insert in layers_to_insert:
                    layer_to_insert.name += "_insert"
                    mylogger.info(layer_to_insert)
                    new_model.add(layer_to_insert)
        new_model.build(MLA_model.input_shape)
    else:
        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output.shape.as_list()
            new_layers = layer_matching.layer_concats[layer_name_to_insert](output_shape)
            for l in new_layers:
                l.name += "_insert"
                mylogger.info('insert layer {}'.format(str(l)))
                x = l(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(MLA_model, operation={MLA_model.layers[layer_index_to_insert].name: layer_addition})

    tuples = []
    import time
    start_time = time.time()

    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_MLA'):
            key = layer_name[:-9]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    end_time = time.time()
    print('set weight cost {}'.format(end_time - start_time))

    return new_model

def insert_node(table, index, new_node_name, **kwargs):
    '''
    :param table:
    :param index:
    :param new_node_name:
    :param kwargs:
    :return:
    '''
    target_node = table.nodeList[index]
    # if the module num is more than mindsporeModel, we need copy the module
    if len(target_node.node_module) > 1:
        table, model_ast = copy_module(table=table, index=index, param_dict=param_dict)

    # insert the node after the index
    layer_utils = LayerUtils()
    if new_node_name in layer_utils.available_model_level_layers.keys():
        insert_str = layer_utils.available_model_level_layers[new_node_name](**kwargs)
        # get op_name
        op_name = "addNode_" + table.add_node_num
        out_name = "opt_addNode_" + table.add_node_num
        table.add_node_num += 1
        # get full inser str
        insert_str = op_name + " = " + insert_str
        insert_node = ast.parse(insert_str).body[0]

        #insert the node
        # in ast, direct add node after the statement
        for module in table.ast.body:
            if isinstance(module, ast.ClassDef) and module.name == target_node.node_module[-1]:
                assert len(target_node.input_list) == 1
                ast_index = target_node.ast_index[-1]
                init_func = module.body[0]
                init_func.body.insert(ast_index[0], insert_node)
        node_len = table.node_list_len()
        insert_index = node_len
        #默认插入的module位置和上一行相同
    else:
        print("{} not implemented!".format(new_node_name))

if __name__ == "__main__":
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    model_ast = astor.parse_file(model_path)
    