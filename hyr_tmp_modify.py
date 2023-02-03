import mindspore
import numpy as np

class ActivationUtils:
    def __init__(self):
        self.available_model_level_layers = {}
        self.is_input_legal = {}
        self.available_model_level_layers['celu_layer'] = ActivationUtils.celu_layer
        self.is_input_legal['celu_layer'] = ActivationUtils.celu_layer_input_legal
        self.available_model_level_layers['elu_layer'] = ActivationUtils.elu_layer
        self.is_input_legal['elu_layer'] = ActivationUtils.elu_layer_input_legal
        self.available_model_level_layers['fastgelu_layer'] = ActivationUtils.fastgelu_layer
        self.is_input_legal['fastgelu_layer'] = ActivationUtils.fastgelu_layer_input_legal
        self.available_model_level_layers['gelu_layer'] = ActivationUtils.gelu_layer
        self.is_input_legal['gelu_layer'] = ActivationUtils.gelu_layer_input_legal
        self.available_model_level_layers['HShrink_layer'] = ActivationUtils.HShrink_layer 
        self.is_input_legal['HShrink_layer'] = ActivationUtils.HShrink_layer_input_legal
        self.available_model_level_layers['HSigmoid_layer'] = ActivationUtils.HSigmoid_layer
        self.is_input_legal['HSigmoid_layer'] = ActivationUtils.HSigmoid_layer_input_legal
        self.available_model_level_layers['HSwish_layer'] = ActivationUtils.HSwish_layer
        self.is_input_legal['HSwish_layer'] = ActivationUtils.HSwish_layer_input_legal
        self.available_model_level_layers['leaky_relu_layer'] = ActivationUtils.leaky_relu_layer
        self.is_input_legal['leaky_relu_layer'] = ActivationUtils.leaky_relu_layer_input_legal
        self.available_model_level_layers['LogSigmoid_layer'] = ActivationUtils.LogSigmoid_layer
        self.is_input_legal['LogSigmoid_layer'] = ActivationUtils.LogSigmoid_layer_input_legal
        self.available_model_level_layers['LogSoftmax_layer'] = ActivationUtils.LogSoftmax_layer
        self.is_input_legal['LogSoftmax_layer'] = ActivationUtils.LogSoftmax_layer_input_legal
        self.available_model_level_layers['prelu_layer'] = ActivationUtils.prelu_layer
        self.is_input_legal['prelu_layer'] = ActivationUtils.prelu_layer_input_legal
        self.available_model_level_layers['relu_layer'] = ActivationUtils.relu_layer
        self.is_input_legal['relu_layer'] = ActivationUtils.relu_layer_input_legal
        self.available_model_level_layers['relu6_layer'] = ActivationUtils.relu6_layer
        self.is_input_legal['relu6_layer'] = ActivationUtils.relu6_layer_input_legal
        self.available_model_level_layers['Sigmoid_layer'] = ActivationUtils.Sigmoid_layer
        self.is_input_legal['Sigmoid_layer'] = ActivationUtils.Sigmoid_layer_input_legal
        self.available_model_level_layers['softmax_layer'] = ActivationUtils.softmax_layer
        self.is_input_legal['softmax_layer'] = ActivationUtils.softmax_layer_input_legal
        self.available_model_level_layers['softshrink_layer'] = ActivationUtils.softshrink_layer
        self.is_input_legal['softshrink_layer'] = ActivationUtils.softshrink_layer_input_legal
        self.available_model_level_layers['tanh_layer'] = ActivationUtils.tanh_layer
        self.is_input_legal['tanh_layer'] = ActivationUtils.tanh_layer_input_legal
        # self.available_model_level_layers['no_activation'] = ActivationUtils.no_activation_layer
        

    @staticmethod
    def celu_layer(input_shape, alpha=1.0):
        layer_str = "nn.CELU(alpha={})".format(alpha)
        return [layer_str]
    
    @staticmethod
    def celu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def elu_layer(alpha = 1.0):
        layer_str = "nn.ELU(alpha = {})".format(alpha)
        return [layer_str]
    
    @staticmethod
    def elu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def fastgelu_layer(input_shape):
        layer_str = "nn.FastGelu()".format()
        return [layer_str]
    
    @staticmethod
    def fastgelu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def gelu_layer(input_shape, approximate=True):
        layer_str = "nn.GELU(approximate={})".format(approximate)
        return [layer_str]
    
    @staticmethod
    def gelu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def HShrink_layer(input_shape, lambd=0.5):
        layer_str = "nn.HShrink(lambd={})".format(lambd)
        return [layer_str]
    
    @staticmethod
    def HShrink_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def HSigmoid_layer(input_shape):
        layer_str = "nn.HSigmoid()".format()
        return [layer_str]
    
    @staticmethod
    def HSigmoid_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def HSwish_layer(input_shape):
        layer_str = "nn.HSwish()".format()
        return [layer_str]
    
    @staticmethod
    def HSwish_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def leaky_relu_layer(alpha=0.2):
        layer_str = "nn.LeakyReLU(alpha={})".format(alpha)

        return [layer_str]

    @staticmethod
    def leaky_relu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def LogSigmoid_layer(input_shape):
        layer_str = "nn.LogSigmoid()".format()
        return [layer_str]
    
    @staticmethod
    def LogSigmoid_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def LogSoftmax_layer(input_shape, axis=-1):
        layer_str = "nn.LogSoftmax(axis={})".format(axis)
        return [layer_str]
    
    @staticmethod
    def LogSoftmax_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def prelu_layer(input_shape):
        # PReLU(xi)=max(0,xi)+w∗min(0,xi)
        layer_str = "nn.PReLU()".format()
        # channel参数：可以是int，值是1或输入Tensor x 的通道数。默认值：1
        # w 默认值：0.25
        return layer_str
    
    @staticmethod
    def prelu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def relu_layer(input_shape):
        layer_str = "nn.ReLU()".format()
        return [layer_str]

    @staticmethod
    def relu_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def relu6_layer(input_shape):
        layer_str = "nn.ReLU6()".format()
        return [layer_str]
    
    @staticmethod
    def relu6_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def Sigmoid_layer(input_shape):
        layer_str = "nn.Sigmoid()"
        return [layer_str]
    
    @staticmethod
    def Sigmoid_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def softmax_layer(input_shape, axis=-1):
        layer_str = "nn.Softmax(axis={})".format(axis)
        return [layer_str]
        
    @staticmethod
    def softmax_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def softshrink_layer(input_shape, lambd=0.5):
        layer_str = "nn.SoftShrink(lambd={})".format(lambd)
        return [layer_str]
    
    @staticmethod
    def softshrink_layer_input_legal(input_shape):
        return True
    
    @staticmethod
    def tanh_layer(input_shape):
        layer_str = "nn.Tanh()"
        return [layer_str]
    
    @staticmethod
    def tanh_layer_input_legal(input_shape):
        return True

    def pick_act_randomly(self, activations = None):
        if activations is None:
            avaliable_acts = [item for item in self.available_model_level_layers.keys()]
        else:
            avaliable_acts = activations
        index = np.random.randint(0, len(avaliable_acts))
        act_str_function = self.available_model_level_layers[avaliable_acts[index]]
        input_shape = None
        act_str = act_str_function(input_shape)
        return act_str[0]

if __name__ == "__main__":
    act_utils = ActivationUtils()
    new_activations=None
    act_str= act_utils.pick_act_randomly(new_activations)
    print(act_str)
    
    
    