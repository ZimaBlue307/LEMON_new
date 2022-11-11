import json
import ast
import astunparse
import astor
from mindspore import Tensor

from scripts.mutation.model_mutation_operators import *

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
                if isinstance(sub_node, ast.Assign) and isinstance(sub_node.value, ast.Call): # check the assign node is using a operator, not just a parameter
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
                        pre_stat = "save.append(['{}', '{} start'])".format(target, target)
                        aft_stat = "save.append(['{}', '{} end'])".format(target, target)
                        pre_node = ast.parse(pre_stat).body[0]
                        aft_node = ast.parse(aft_stat).body[0]
                        node.body.insert(i, pre_node)
                        node.body.insert(i+2, aft_node)
                        # print(astunparse.unparse(node))
                        i = i+1
                    else:
                        # get shape
                        print_str = "save.append(['{}', {}.shape])".format(target, target)
                        # sub_node_str = astunparse.unparse(sub_node)
                        # print_str = sub_node_str  + print_str + '\n'
                        print_ast = ast.parse(print_str)
                        print_node = print_ast.body[0]
                        print_ast_str = ast.dump(print_node)
                        node.body.insert(i+1, print_node)
                        # i += 1
                        # print(print_ast_str)

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


class Shape_uitls(object):

    def __init__(self):
        super(Shape_uitls, self).__init__()

    @staticmethod
    def modify_ast(source_ast, output_path):
        '''
        Given the ast of source code, and theoutput path, modify source code and save it
        :param source_ast:
        :param output_path:
        :return:
        '''

        # add a list to save the shape
        i = 0
        for i in range(len(source_ast.body)):
            if not isinstance(source_ast.body[i], ast.Import) and not isinstance(source_ast.body[i], ast.ImportFrom):
                break
        list_str = 'save = list()'
        list_node = ast.parse(list_str).body[0]
        source_ast.body.insert(i, list_node)

        # add import json
        json_node = ast.parse('import json').body[0]
        source_ast.body.insert(i, json_node)

        visitor = PrintVisitor()
        visitor.visit(source_ast)

        # add print in the last construct function
        last_construct = source_ast.body[-1].body[-1]
        # print(astunparse.unparse(last_construct))
        print_str1 = "with open('{}', 'w') as f:\n   json.dump(save, f)".format(output_path)
        print_node = ast.parse(print_str1)
        print_node1 = print_node.body[0]
        # print_node2 = print_node.body[1]

        last_construct.body.insert(-1, print_node1)
        # last_construct.body.insert(-1, print_node2)

        # print_str = "print(save)"
        # print_node = ast.parse(print_str).body[0]
        # last_construct.body.insert(-1, print_node)
        return source_ast

    def module_analyze(self, shape_list, target_list, module_name, index):
        while index < len(shape_list):
            item = shape_list[index]
            if 'module' in item[0]:
                if 'start' in item[1]:
                    tmp_name = module_name + '.' + item[0]
                    target_list, index = self.module_analyze(shape_list, target_list, tmp_name, index + 1)
                elif 'end' in item[1]:
                    print('module finished! index come to {}'.format(index))
                    return target_list, index
            else:
                op_name = module_name + '.' + item[0]
                target_list.append([op_name, item[1]])
            index += 1
        print('module finished! index come to {}'.format(index))
        return target_list, index
