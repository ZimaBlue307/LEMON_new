import ast
import astor
import astunparse
import os

def generate_ast(python_file_path):
    # file_path such as XXX/resnet20_cifar100_origin
    python_name = tuple(python_file_path.split('/'))[-1]
    file_name = python_file_path + "/" + python_name + ".py"
    python_file_ast = astor.parse_file(file_name)# 将 Python 文件解析为 AST
    
    # save ast string in a file, for test and check
    ast_string = astunparse.dump(python_file_ast)
    file_store_ast = open('ast_example.py', mode = 'w+')
    file_store_ast.write(ast_string)
    
    # return an ast variable
    return python_file_ast

# 仅适用于各个class的construct
def build_network_graph_cons(ast_varible, ClassDef_name, function_name, graph_dict):
    """
    usually, ClassDef_name = "MindSporeModel"
             function_name = "construct"
    input: an ast variable, ClassDef_name, and general_list that store the entire structure of ast variable
    return: general_dict with key_value_pair: ClassDef_name, general_list
            a general_list holds the structure of the ClassDef_name network graph.
            general_list = [sublist, sublist, ......] 
            sublist: [assign_input, assign_op, assign_output, assign_other] 
                assign_input: a list with several input_Tensor nodes
                assign_op: corresponding operator node
                assign_output: output_Tensor node
                assign_other: other parameters of input_Tensor, such as nodes of type Tuple, Attribute; list
    """
    dict_key = ClassDef_name
    if dict_key not in graph_dict.keys():
        general_list = []
        for level1_node in ast_varible.body:
            if isinstance(level1_node, ast.ClassDef) and level1_node.name == ClassDef_name:#MindSporeModel
                    for level2_node in level1_node.body:
                        if isinstance(level2_node, ast.FunctionDef) and level2_node.name == function_name: #construct
                            for level3_node in level2_node.body:
                                if isinstance(level3_node, ast.Assign):#construct中的Assign node
                                    sublist = []
                                    assign_output = None
                                    assign_input = None
                                    assign_op = None
                                    for level4_node in level3_node.targets:# Assign node中的targets
                                        if isinstance(level4_node, ast.Name):
                                            assign_output = level4_node #such as opt_transpose_0,输出张量的node
                                        if isinstance(level4_node, ast.Attribute):
                                            assign_output = level4_node # 在init函数中会进入此分支
                                    for level4_node in ast.walk(level3_node.value):# Assign node中的value
                                        if isinstance(level4_node, ast.Call):#Assign node中的value中的Call
                                            # for level5_node in ast.walk(level4_node.func):
                                            #     if isinstance(level5_node, ast.Attribute):# Call的Attribute
                                            assign_op = level4_node# 整个operator的node
                                            for level5_node in level4_node.args:
                                                assign_input = []
                                                if isinstance(level5_node, ast.Name):
                                                    assign_input.append(level5_node) # 输入张量node
                                        else: 
                                            break
                                    if assign_op is None:
                                        continue
                                    sublist = [assign_input, assign_op, assign_output]
                                    general_list.append(sublist)
        graph_dict[dict_key] = general_list
    return graph_dict                                                       

#刚从build_network_graph_cons复制，还没修改
def build_network_graph_init(ast_varible, ClassDef_name, function_name, graph_dict):
    """
    function_name: __init__
    """
    dict_key = ClassDef_name
    if dict_key not in graph_dict.keys():
        general_list = []
        for level1_node in ast_varible.body:
            if isinstance(level1_node, ast.ClassDef) and level1_node.name == ClassDef_name:#MindSporeModel
                    for level2_node in level1_node.body:
                        if isinstance(level2_node, ast.FunctionDef) and level2_node.name == function_name: #construct
                            for level3_node in level2_node.body:
                                if isinstance(level3_node, ast.Assign):#construct中的Assign node
                                    sublist = []
                                    assign_output = None
                                    assign_input = None
                                    assign_op = None
                                    for level4_node in level3_node.targets:# Assign node中的targets
                                        if isinstance(level4_node, ast.Name):
                                            assign_output = level4_node #such as opt_transpose_0,输出张量的node
                                        if isinstance(level4_node, ast.Attribute):
                                            assign_output = level4_node # 在init函数中会进入此分支
                                    for level4_node in ast.walk(level3_node.value):# Assign node中的value
                                        if isinstance(level4_node, ast.Call):#Assign node中的value中的Call
                                            # for level5_node in ast.walk(level4_node.func):
                                            #     if isinstance(level5_node, ast.Attribute):# Call的Attribute
                                            assign_op = level4_node# 整个operator的node
                                            for level5_node in level4_node.args:
                                                assign_input = []
                                                if isinstance(level5_node, ast.Name):
                                                    assign_input.append(level5_node) # 输入张量node
                                        else: 
                                            break
                                    if assign_op is None:
                                        continue
                                    sublist = [assign_input, assign_op, assign_output]
                                    general_list.append(sublist)
        graph_dict[dict_key] = general_list
    return graph_dict
                                 
def whether_nested(op_name):
    if 'module' in op_name:
        # print("operator {} has nested structures. ".format(op_name))
        return True
    else:
        # print("operator {} does not have nested structures. ".format(op_name))
        return False

def handle_nested_cons(ast_varible, assign_op, graph_dict): #处理嵌套结构
    """
    input: ast_varible, ast tree;
           assign_op, operator_name;
    output: similar with build_network_graph. but will add a new key_value pair in general_dict
    """
    operator_name = assign_op.func.attr
    nest_name = tuple(operator_name.split("_"))[0] # such as module2
    Module_name = nest_name.capitalize() # such as Module2
    function_name = "construct"
    build_network_graph_cons(ast_varible, Module_name, function_name, graph_dict)
    Module_value = graph_dict[Module_name]
    for sublist in Module_value:
        sub_op = sublist[1]
        if whether_nested(sub_op.func.attr):
            handle_nested_cons(ast_varible, sub_op, graph_dict)
    return graph_dict
        
def ast2code_cons(general_dict, construct_str):
    Mindsporemodel = general_dict["MindSporeModel"]
    for sublist in Mindsporemodel:
        output_tensor = sublist[2]
        output_tensor_name = output_tensor.id
        operator_and_input = sublist[1]
        operator_and_input_str = astor.to_source(operator_and_input, indent_with=' ' * 4, add_line_information=False, source_generator_class=astor.SourceGenerator)
        assign_str = "{} = {}\n".format(output_tensor_name, operator_and_input_str)
        construct_str += assign_str
    return construct_str
    
                            
if __name__ == "__main__":
    python_file_path = "alexnet_cifar10_origin"
    python_file_ast = generate_ast(python_file_path)
    
    ClassDef_name = "MindSporeModel"
    function_name_construct = "construct"
    graph_dict_construct = {}
    graph_dict_construct = build_network_graph_cons(python_file_ast, ClassDef_name, function_name_construct, graph_dict_construct)
    
    ms_cons_value = graph_dict_construct["MindSporeModel"]
    for sublist in ms_cons_value:
        operator = sublist[1]
        operator_name = operator.func.attr
        if whether_nested(operator_name):
            handle_nested_cons(python_file_ast, operator, graph_dict_construct)
    # remove就是直接删除ms_cons_value中的某个sublist就好了
    # copy就挑一个list在construct里加一个就好了
    copy_index = 4
    count = 0
    for sublist in ms_cons_value:
        count += 1
        if count == copy_index:
            copy_list = ms_cons_value[3]
        
            
        
        
        

    # construct_str = ""
    # construct_str = ast2code_cons(graph_dict_construct, construct_str)
    # print(construct_str)
    # print("================")
    
    #对于MindSporeModel的value中的sublist中的assign_op判断和处理它的嵌套结构
    # MindSporeModel_value_cons = general_dict_construct["MindSporeModel"]
    # for sublist in MindSporeModel_value_cons:
    #         operator_name = sublist[1].func.attr
    #         if whether_nested(operator_name):
    #             handle_nested_cons(python_file_ast, operator_name, function_name_construct, general_dict_construct) 

    # construct_str = ""
    # construct_str = ast2code(general_dict_construct, construct_str)