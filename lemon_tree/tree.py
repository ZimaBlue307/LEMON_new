import inspect
import os
import ast
import astunparse
import copy
import sys
import astor
# sys.path.append('..')
# from origin_model.ms_model.resnet20_cifar100.resnet20_cifar100_origin import MindSporeModel

import mindspore


class nodeInfo(object):

    '''
    node is a representation of operators, it contains:
    cell function or operator function
    '''

    def __init__(self, name, before=None, after=None, lineno=None):
        self.name = name
        self.before = before
        self.after = after
        self.lineno = lineno
        self.sub_list = None
        self.is_return=False

    def set_input(self, input: list):
        self.before = copy.deepcopy(input)

    def set_output(self, output: list):
        self.after = copy.deepcopy(output)

    # manage sub node list is current node is a module
    def add_sub_node(self, sub_node):
        if self.sub_list is None:
            self.sub_list = dict()
        self.sub_list[sub_node.name] = sub_node

    def set_end_node(self):
        self.is_return = True

class Value(object):
    def __init__(self, name):
        self.name = name
        self.before = None
        self.after = None

    def set_input(self, input):
        self.before = input

    def set_output(self, output):
        self.after = output


def get_name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Name):
        return node.id
    else:
        print("current node type {} does not need type info".format(type(node)))
        return None


class ClassNode(object):
    def __init__(self, node, nodelist, name):
        self.scope = node
        self.name = name
        self.valueList = dict() # save the index in the body of ast
        self.nodeList = nodelist

    def update_valuelist(self, valueList):
        self.valueList = valueList


def get_code(code_ast):
    '''

    :param code_ast:
    :return: a list of nodeInfo, each node include the name and the link direction
    '''

    class initVisitor(ast.NodeVisitor):
        # use init function to construct the list of nodeInfo
        def __init__(self):
            super(initVisitor, self).__init__()
            self.nodeList = dict()

        def visit_Assign(self, node: ast.Assign):
            if len(node.targets) > 1:
                raise RuntimeError("Do not support multiply outputs in network")
            if isinstance(node.value, ast.Call):
                tmp = None
                if isinstance(node.value.func, ast.Name): ## module
                    tmp = nodeInfo(node.value.func.id)
                elif isinstance(node.value.func, ast.Attribute): # mindspore nn class
                    tmp = nodeInfo(node.value.func.attr)
                if not node.targets[0].attr in self.nodeList.keys():
                    self.nodeList[node.targets[0].attr] = tmp
            return self.generic_visit(node)

        def get_node_list(self, init_node):
            self.generic_visit(init_node)
            return self.nodeList

    class constructVisitor(ast.NodeVisitor):

        def __init__(self, nodeList):
            super(constructVisitor, self).__init__()
            self.nodeList = nodeList
            self.valueList = dict()

        def visit_Assign(self, node):
            value = Value(node.targets[0].id)
            # if it is a call
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    input_name = node.value.func.attr
                    value.set_input(self.find_node(input_name))

                    args = node.value.args
                    inputs = list()
                    for arg in args:
                        # for each arg, it means a input
                        if isinstance(arg, ast.Name):
                            pass
                else:
                    print("node {} is non-Attribute, please check it".format(node.value))
                    pass

            elif isinstance(node.value, ast.BinOp):
                left_name, right_name = get_name(node.value.left), get_name(node.value.right)

            self.valueList[node.targets[0].id] = value
            return self.generic_visit(node)

        def visit_Return(self, node: ast.Return):
            return_name = node.value.id
            return_node = self.find_node(return_name)
            if isinstance(return_node, Value):
                # may be before node is multiple?
                return_node.before.set_end_node()
            return self.generic_visit(node)

        def find_node(self, name):
            # find node having the name
            for key, item in enumerate(self.nodeList):
                if key == name:
                    return item
            for key, item in enumerate(self.valueList):
                if key == name:
                    return item


        def add_direction(self, construct_node):
            self.generic_visit(construct_node)
            return self.nodeList

    result_code_list = dict()
    init_function, construct_function = None, None
    for item in code_ast.body:
        if isinstance(item, ast.ClassDef):
            # print(item.body)

            for ele in item.body:
                if isinstance(ele, ast.FunctionDef):
                    if ele.name == 'construct':
                        construct_function = ele
                    elif ele.name == '__init__':
                        init_function = ele

            init_visitor = initVisitor()
            module_code_list = init_visitor.get_node_list(init_function)
            cons_visitor = constructVisitor(module_code_list)
            item_node = ClassNode(node=item, nodelist=module_code_list, name=item.name)
            # module_code_list = cons_visitor.add_direction(construct_function)
            result_code_list[item.name] = item_node

    return result_code_list

if __name__ == "__main__":

    # net = MindSporeModel()
    # param_path = f'../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.ckpt'
    # param = mindspore.load_checkpoint(param_path)
    # mindspore.load_param_into_net(net, param)

    # get python file
    model_path = f"../origin_model/ms_model/resnet20_cifar100/resnet20_cifar100_origin.py"
    model_ast = astor.parse_file(model_path)
    result_dict = get_code(model_ast)
