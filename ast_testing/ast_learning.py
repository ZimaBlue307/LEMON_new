import ast
import astor
import astunparse
import graphviz
from graphviz import Digraph

#可视化ast
def visualize_ast(node, nodes, pindex, g):
    name = str(type(node).__name__)
    index = len(nodes)
    nodes.append(index)
    g.node(str(index), name)
    if index != pindex:
        g.edge(str(index), str(pindex))
    for n in ast.iter_child_nodes(node):
        visualize_ast(n, nodes, index, g)

#遍历与寻找
class myNodeVisitor(ast.NodeVisitor):
    
    def __init__(self):
        pass
    
    def visit_Name(self, node): # 访问指定的名称
        # print(type(node.id))
        if node.id == "MindSporeModel":
            print("=====find the node MindSporeModel !=====")
            return True
        self.generic_visit(node) # 遍历子节点
    
    def visit_FunctionDef(self, node):
        if node.name == "construct" :
            print("=====find the function construct !=====")
            return True
        self.generic_visit(node) # 遍历子节点
            
    def new_function1(self, node):
        if self.visit_Name(node):
            if self.visit_FunctionDef(node):
                print("good")
        
        

if __name__ == "__main__":
    open_file_dir = "resnet20_cifar100_origin/resnet20_cifar100_origin.py"
    with open(open_file_dir) as f:
        content = f.read()
    content_ast = astor.parse_file(open_file_dir)# 将 Python 文件解析为 AST
    # content_ast = ast.parse(content)
    ast_string = astunparse.dump(content_ast)
    # print(astunparse.dump(content_ast))
    file_store_ast = open('ast_string.py', mode = 'w+')
    file_store_ast.write(ast_string)
    #目前1493行是class MindSporeModel
    
    # #可视化整个ast文件
    # with open(open_file_dir) as f:
    #     graph = Digraph(format="png")
    #     tree = ast.parse(f.read())
    #     visualize_ast(tree, [], 0, graph)
    #     graph.render("test")
    
    #即使在模型结构的python文件中加入张量定义和网络输入，也无法执行。
    # result = compile(content, '<string>', 'exec')
    # exec(result)
    visitor_ast = myNodeVisitor()
    visitor_ast.visit(content_ast)
    
    
    
    
        
    
    