import argparse
import copy
from ctypes import util
import sys
from operator import truediv
from scripts.mutation.mutation_utils import No_Activation
import mindspore
from mindspore.rewrite import *
import numpy as np
from scripts.tools import utils
import math
from typing import *
from scripts.mutation.mutation_utils import *
from scripts.mutation.layer_matching import LayerMatching
import random
import os
import warnings
from scripts.logger.lemon_logger import Logger
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_global_var(model):
    model_tree = mindspore.rewrite.SymbolTree.create(model)
    global_vars = model_tree._symbol_tree._global_vars
    return global_vars
