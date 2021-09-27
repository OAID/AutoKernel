import logging
import torch
from torch import nn
import numpy as np

class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, key_name):
        """Decorator to register a function or class."""
        def add(key, value):
            self[key] = value
            return value
        # @reg.register('alias')
        return lambda func: add(key_name, func)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

op_reg = Register("op_register")
@op_reg.register("Conv_Add_fused")
def run_Conv_Add_fused(node):
    inp_0 = node.input[0]
    inp_1 = node.input[1]
    inp_2 = node.input[2]
    inp_0_tensor = torch.tensor(np.array(inp_0.value,dtype=np.float32).reshape(inp_0.dims))
    inp_1_tensor = torch.tensor(np.array(inp_1.value,dtype=np.float32).reshape(inp_1.dims))
    inp_2_tensor = torch.tensor(np.array(inp_2.value,dtype=np.float32).reshape(inp_2.dims))
    param_0 = node.attr
    conv_0 = nn.Conv2d(param_0.c_in, param_0.c_out, param_0.ksize, param_0.stride, param_0.pad,1,1,True)
    conv_0.weight.data = inp_1_tensor
    conv_0.bias.data = inp_2_tensor
    tmp_0 = conv_0(inp_0_tensor)
    out = tmp_0.detach().numpy()
    out_0 = node.output[0]
    out_0.value=out
    if out_0.reshaped==0:
        out_0.dims=out.shape
@op_reg.register("Relu")
def run_Relu(node):
    inp_0 = node.input[0]
    inp_0_tensor = torch.tensor(np.array(inp_0.value,dtype=np.float32).reshape(inp_0.dims))
    relu_0 = nn.ReLU()
    tmp_0 = relu_0(inp_0_tensor)
    out = np.array(tmp_0)
    out_0 = node.output[0]
    out_0.value=out
    if out_0.reshaped==0:
        out_0.dims=out.shape
@op_reg.register("MaxPool")
def run_MaxPool(node):
    inp_0 = node.input[0]
    inp_0_tensor = torch.tensor(np.array(inp_0.value,dtype=np.float32).reshape(inp_0.dims))
    param_0 = node.attr
    maxpool_0 = nn.MaxPool2d(param_0.ksize, param_0.stride, param_0.pad)
    tmp_0 = maxpool_0(inp_0_tensor)
    out = np.array(tmp_0)
    out_0 = node.output[0]
    out_0.value=out
    if out_0.reshaped==0:
        out_0.dims=out.shape
@op_reg.register("MatMul_Add_fused")
def run_MatMul_Add_fused(node):
    inp_0 = node.input[0]
    inp_1 = node.input[1]
    inp_2 = node.input[2]
    inp_0_tensor = torch.tensor(np.array(inp_0.value,dtype=np.float32).reshape(inp_0.dims))
    inp_1_tensor = torch.tensor(np.array(inp_1.value,dtype=np.float32).reshape(inp_1.dims))
    inp_2_tensor = torch.tensor(np.array(inp_2.value,dtype=np.float32).reshape(inp_2.dims))
    tmp_0 = torch.matmul(inp_0_tensor,inp_1_tensor)
    tmp_1 = torch.add(tmp_0,inp_2_tensor)
    out = np.array(tmp_1)
    out_0 = node.output[0]
    out_0.value=out
    if out_0.reshaped==0:
        out_0.dims=out.shape
