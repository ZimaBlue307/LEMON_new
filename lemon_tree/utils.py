import ast
import astunparse


BinOpTable = {
    "<class '_ast.Add'>": "add",
    "class '_ast.Sub'": "sub",
    "class '_ast.Mult'": "mult",
    "class '_ast.Div'": "div",
    "class '_ast.FloorDiv'": "floordiv",
    "class '_ast.Mod'": "mod",
    "class '_ast.Pow'": "pow",
    "class '_ast.LShift'": "lshift",
    "class '_ast.RShift'": "rshift",
    "class '_ast.BitOr'": "bitor",
    "class '_ast.BitXor'": "bitxor",
    "class '_ast.BitAnd'": "bitand",
    "class '_ast.MatMult'": "matmult"
}


class Node(object):
    def __init__(self, index, unique_name, shape,
                 operator_name = None, node_module=None, input_list=None,
                 output_list=None, output_name=None, copy_num=None,
                 ast_index = None, ms_class = None):
        self.index = index
        self.unique_name = unique_name # module name + output name
        self.operator_name = operator_name
        self.node_module = node_module
        self.input_list = input_list
        self.output_list = output_list
        self.shape = shape
        self.output_name = output_name
        self.copy_num = copy_num
        self.ast_index = ast_index
        self.ms_class = ms_class #newly added by hyr

    def set_uniquename(self, unique_name):
        self.unique_name = unique_name

    def set_operator_name(self, operator_name):
        self.operator_name = operator_name

    def set_input(self, input_list):
        self.input_list = input_list

    def set_output(self, output_list):
        self.output_list = output_list

    def set_module(self, module):
        self.node_module = module

    def set_copy_num(self, copy_num):
        self.copy_num = copy_num

    def set_ast_index(self, ast_index):
        self.ast_index = ast_index

    def set_ms_operator(self, ms_class): #newly added by hyr
        self.ms_class = ms_class
        
        
    def get_prefix(self):
        unique_name \
            = self.unique_name
        prefix = unique_name.split(".")
        prefix = '.'.join(prefix[:-1])
        return prefix

class Table(object):
    def __init__(self, model_ast):
        self.nodeList = dict()
        self.ast = model_ast
        self.add_node_num = 0

    def add_node(self, node):
        self.nodeList[node.index] = node

    def print(self):
        for item in self.nodeList:
            item = self.nodeList[item]
            print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module, '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list, "+++++", item.copy_num, "+++++", item.ast_index)

    def print_nodelist(self, indices):
        keys = self.nodeList.keys()
        if isinstance(indices, list):
            for index in indices:
                item = self.nodeList[index]
                print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module,
                      '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list,
                      "+++++", item.copy_num, "+++++", item.ast_index)
        elif isinstance(indices, int):
            item = self.nodeList[indices]
            print(item.index, '+++++', item.unique_name, '+++++', item.operator_name, '+++++', item.node_module,
                  '+++++', item.output_name, '+++++', item.shape, '+++++', item.input_list, '+++++', item.output_list,
                  "+++++", item.copy_num, "+++++", item.ast_index)

    def node_list_len(self):
        return len(self.nodeList)

    def save_ast(self,  save_path):
        code_str = astunparse.unparse(self.ast)
        with open(save_path, 'w') as f:
            f.write(code_str)