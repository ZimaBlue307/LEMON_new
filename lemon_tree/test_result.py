import os

model_path = 'result.py'

module = __import__(model_path[:-3])
print(module)