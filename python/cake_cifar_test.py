import sys
import time
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.autograd import Function
# import linear_cpp
from torch.utils.cpp_extension import load


# Load training data
transform_train = transforms.Compose([                                   
    transforms.RandomCrop(32, padding=4),                                       
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# Load testing data
transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)
print('Finished loading datasets!')


# linear_cpp = load(name = 'linear_cpp', sources = ['linear.cpp'], verbose=True)

linear_cpp = load(name = 'linear_cpp', 
                sources = ['linear.cpp', 'block_sizing.cpp', 
                            'cake_dgemm.cpp', 'pack.cpp', 'util.cpp', 'unpack.cpp'],
                extra_cflags = ['-O3','-Wall','-Wno-unused-function','-Wfatal-errors', '-fopenmp',
                                '-fPIC',' -D_POSIX_C_SOURCE=200112L', '-DBLIS_VERSION_STRING=\"0.8.0-13\"'],
                extra_ldflags = ['/usr/local/lib/libblis.a', '-lm', '-lpthread', '-lrt' ],
                extra_include_paths = ['.', '/usr/local/include/blis'], 
                verbose=True)


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





# import torch
# import torch.nn as nn
# from torch.utils.cpp_extension import load


# cake_lib = load(name = 'cake_linear', 
#                 sources = ['cake_linear.cpp', 'block_sizing.cpp ', 
#                             'cake_dgemm.cpp', 'pack.cpp', 'util.cpp', 'unpack.cpp'],
#                 extra_cflags = ['-O3','-Wall','-Wno-unused-function','-Wfatal-errors', '-fopenmp',
#                                 '-fPIC',' -D_POSIX_C_SOURCE=200112L', '-DBLIS_VERSION_STRING=\"0.8.0-13\"'],
#                 extra_ldflags = ['/usr/local/lib/libblis.a', '-lm', '-lpthread', '-lrt' ],
#                 extra_include_paths = ['.', '/usr/local/include/blis'], 
#                 verbose=True)


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



def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    '''
    A nn.Sequential layer executes its arguments in sequential order. In
    this case, it performs Conv2d -> BatchNorm2d -> ReLU. This is a typical
    block of layers used in Convolutional Neural Networks (CNNs). The 
    ConvNet implementation below stacks multiple instances of this three layer
    pattern in order to achieve over 90% classification accuracy on CIFAR-10.
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class ConvNet(nn.Module):
    '''
    A 9 layer CNN using the conv_block function above. Again, we use a
    nn.Sequential layer to build the entire model. The Conv2d layers get
    progressively larger (more filters) as the model gets deeper. This 
    corresponds to spatial resolution getting smaller (via the stride=2 blocks),
    going from 32x32 -> 16x16 -> 8x8. The nn.AdaptiveAvgPool2d layer at the end
    of the model reduces the spatial resolution from 8x8 to 1x1 using a simple
    average across all the pixels in each channel. This is then fed to the 
    single fully connected (linear) layer called classifier, which is the output
    prediction of the model.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1)
            )
        #
        # self.classifier = nn.Linear(256, 10)
        self.classifier = cake_linear(256, 10)
        #
    def forward(self, x):
        '''
        The forward function is called automatically by the model when it is
        given an input image. It first applies the 8 convolution layers, then
        finally the single classifier layer.
        '''
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)

 

torch.manual_seed(43) # to give stable randomness

# tracks the highest accuracy observed so far
best_acc = 0

def moving_average(a, n=100):
    '''Helper function used for visualization'''
    ret = torch.cumsum(torch.Tensor(a), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(epoch, train_loss_tracker, train_acc_tracker):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # update optimizer state
        optimizer.step()
        # compute average loss
        train_loss += loss.item()
        train_loss_tracker.append(loss.item())
        loss = train_loss / (batch_idx + 1)
        # compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        # Print status
        sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +  
                         f'| Train Acc: {acc:.3f}')
        sys.stdout.flush()
        # sys.exit(1)
    train_acc_tracker.append(acc)
    sys.stdout.flush()

def test(epoch, test_loss_tracker, test_acc_tracker):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
#
            test_loss += loss.item()
            test_loss_tracker.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
#
            loss = test_loss / (batch_idx + 1)
            acc = 100.* correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()
    #
    # Save checkpoint.
    acc = 100.*correct/total
    test_acc_tracker.append(acc)
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# device = 'cuda'
device = 'cpu'
net = ConvNet()
net = net.to(device)

# PART 1.1: set the learning rate (lr) used in the optimizer.
# Best lr = 0.1
lr = 0.1

# PART 1.2: Change the milestones used in the scheduler to decrease every 25
#           epochs for a total of 4 different lr values over 100 epochs.
# We decrease lr 3 times to get 4 different lr values
milestones = [25, 50, 75]

# PART 1.1: Modify this to train for a short 5 epochs
# PART 1.2: Modify this to train a longer 100 epochs
epochs = 20

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=milestones,
                                                 gamma=0.1)

# Records the training loss and training accuracy during training
train_loss_tracker, train_acc_tracker = [], []

# Records the test loss and test accuracy during training
test_loss_tracker, test_acc_tracker = [], []

print('Training for {} epochs, with learning rate {} and milestones {}'.format(
      epochs, lr, milestones))

start_time = time.time()
for epoch in range(0, epochs):
    train(epoch, train_loss_tracker, train_acc_tracker)
    test(epoch, test_loss_tracker, test_acc_tracker)
    scheduler.step()

total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))

## About 111 seconds, 5 epochs
## About 2167 seconds, 100 epochs

# Plot Train Loss
_train_loss = moving_average(train_loss_tracker)
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(_train_loss)
#plt.xticks(range(0, epochs))
plt.xlabel("batches")
plt.ylabel("loss")
plt.title("Train Loss")

# Plot Test Acc
plt.subplot(1, 2, 2)
plt.plot(test_acc_tracker)
#plt.xlim([0, epoch])
plt.xlabel("epoch")
plt.ylabel("test acc")
plt.title("Test Accuracy")

# Save Figure
# plt.savefig("plot_1-2-1.pdf")
