import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.autograd import Function
import linear_cpp

# linear_cpp = load(name = 'linear_cpp', 
#                 sources = ['cake_pytorch_float/linear.cpp', 'cake_pytorch_float/block_sizing.cpp', 
#                             'cake_pytorch_float/cake_sgemm.cpp', 'cake_pytorch_float/pack.cpp', 'cake_pytorch_float/util.cpp', 'cake_pytorch_float/unpack.cpp'],
#                 extra_cflags = ['-O3','-Wall','-Wno-unused-function','-Wfatal-errors', '-fopenmp',
#                                 '-fPIC',' -D_POSIX_C_SOURCE=200112L', '-DBLIS_VERSION_STRING=\"0.8.0-13\"'],
#                 extra_ldflags = ['/usr/local/lib/libblis.a', '-lm', '-lpthread', '-lrt' ],
#                 extra_include_paths = ['.', '/usr/local/include/blis'], 
#                 verbose=True)

class cake_linear_function(torch.autograd.Function):
    #
    @staticmethod
    def forward(ctx, data, weight, bias=None):
        output = linear_cpp.forward(data, weight, bias)
        ctx.save_for_backward(data, weight, bias, output)
        return output
    #
    @staticmethod
    def backward(ctx, grad_output): 
        data, weight, bias, output = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = linear_cpp.backward(grad_output, data, weight, bias)
        return grad_input, grad_weight, grad_bias
        

class cake_linear(nn.Module):
    #
    def __init__(self, input_features, output_features, bias=True):
        super(cake_linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    #
    def forward(self, data):
        return cake_linear_function.apply(data, self.weight, self.bias)


# get the CPU cache sizes here lscpu


# class cake_linear_function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, data, weight, bias=None):
#         ctx.save_for_backward(data, weight, bias)
#         output = cake_lib.forward(data, weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output
 
#     @staticmethod
#     def backward(ctx, grad_output):
#         data, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#         #
#         if ctx.needs_input_grad[0]:
#             grad_input = cake_lib.backward(grad_output, weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = cake_lib.backward(grad_output.t(), data)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)
#         #
#         return grad_input, grad_weight, grad_bias



# class cake_linear(nn.Module):
#     def __init__(self, input_features, output_features, bias=True):
#         super(cake_linear, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#         #
#         # nn.Parameter is a special kind of Tensor, that will get
#         # automatically registered as Module's parameter once it's assigned
#         # as an attribute. Parameters and buffers need to be registered, or
#         # they won't appear in .parameters() (doesn't apply to buffers), and
#         # won't be converted when e.g. .cuda() is called. You can use
#         # .register_buffer() to register buffers.
#         # nn.Parameters require gradients by default.
#         self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(output_features))
#         else:
#             # You should always register all possible parameters, but the
#             # optional ones can be None if you want.
#             self.register_parameter('bias', None)
#         #
#         # Not a very smart way to initialize weights
#         self.weight.data.uniform_(-0.1, 0.1)
#         if self.bias is not None:
#             self.bias.data.uniform_(-0.1, 0.1)
#         # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
#         # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         # bound = 1 / math.sqrt(fan_in)
#         # nn.init.uniform_(self.bias, -bound, bound)  # bias init
#             #
#     def forward(self, data):
#         # See the autograd section for explanation of what happens here.
#         return cake_linear_function.apply(data, self.weight, self.bias)
#         #
#     def extra_repr(self):
#         # (Optional)Set the extra information about this module. You can test
#         # it by printing an object of this class.
#         return 'input_features={}, output_features={}, bias={}'.format(
#             self.input_features, self.output_features, self.bias is not None
#         )




