import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn import metrics
from elastic_mobilenet import InvertedResidual
from elastic_resnet import ResidualBlock

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def new_forward(bn, mean_est, var_est):
    def lambda_forward(x):

        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)#.mean(4, keepdim=True)  # 1, C, 1, 1
        batch_var = (x - batch_mean) * (x - batch_mean)
        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)#.mean(4, keepdim=True)

        batch_mean = torch.squeeze(batch_mean)
        batch_var = torch.squeeze(batch_var)

        mean_est.update(batch_mean.data, x.size(0))
        var_est.update(batch_var.data, x.size(0))

        # bn forward using calculated mean & var
        _feature_dim = batch_mean.size(0)
        return F.batch_norm(
            x, batch_mean, batch_var, bn.weight[:_feature_dim],
            bn.bias[:_feature_dim], False,
            0.0, bn.eps,
        )
    return lambda_forward

# Modified from https://github.com/mit-han-lab/once-for-all
def calibrate_batchnorm(net, generator, net_type='mobilenet', device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_module = InvertedResidual if net_type == 'mobilenet' else ResidualBlock

    bn_mean = {}
    bn_var = {}
    generator.dataset.set_input_dim(randomize=True, permute_from=[net.input_resolution])
    copy_net = copy.deepcopy(net)

    for block_name, block in copy_net.named_modules():
        if not isinstance(block, block_module):
            continue
        block.bn_calibration_mode = True
        for name, m in block.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # print(name, m)
                name = block_name + '-' + name
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()
                m.forward = new_forward(m, bn_mean[name], bn_var[name])
    copy_net.train()

    with torch.no_grad():
        i = 0
        for volume, _ in generator:
            volume = volume.to(device)
            copy_net(volume)
            i += 1

    del copy_net
    for block_name, block in net.named_modules():

        if not isinstance(block, block_module):
            continue
        for name, m in block.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                name = block_name + '-' + name
                if isinstance(bn_mean[name].avg, int):
                    continue
                feature_dim = bn_mean[name].avg.size(0)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def accuracy_metric_numpy(output, label):
    return np.sum(np.argmax(output, 1) == label) / output.shape[0]

def accuracy_metric(output, label):
    return torch.sum(torch.argmax(output, 1) == label) / output.size(0)

def feature_layer_loss(output, mask, epsilon=1e-5):
    total_loss = 0
    for i in range(len(output)):
        out_channel = output[i].shape[1]
        mask_channel = mask[i].shape[1]
        n_channel = min([out_channel, mask_channel])
        total_loss += F.mse_loss(output[i][:,:n_channel], mask[i][:,:n_channel])
    return total_loss / len(output)

def multiclass_precision(y_pred, y_true, threshold=0.5):
    y_pred = y_pred >= threshold
    precision_score = metrics.precision_score(y_true, y_pred, average='samples')
    return precision_score

def multiclass_recall(y_pred, y_true, threshold=0.5):
    y_pred = y_pred >= threshold
    recall_score = metrics.recall_score(y_true, y_pred, average='samples')
    return recall_score

def multiclass_f1(y_pred, y_true, threshold=0.5):
    y_pred = y_pred >= threshold
    f1_score = metrics.f1_score(y_true, y_pred, average='samples')
    return f1_score


def output_layer_loss(output, teacher_output):
    return F.mse_loss(output, teacher_output)

def calculate_net_size(net, input_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, *input_dim).to(device)
    
    net(dummy_input)

    return net.calculate_param_size() / 1000/ 1000

def calculate_net_feature_size(net, input_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, *input_dim).to(device)
    
    net(dummy_input)

    return net.calculate_feature_size() / 1000/ 1000

def calculate_net_size_from_resnet_config(net):
    net_configuration = copy.deepcopy(net.get_configuration())
    
    depth_configuration = net_configuration['depth_configuration']
    width_configuration = net_configuration['width_configuration']
    width_configuration.insert(0, net.initial_out_channel)

    
    kernels_configuration = net_configuration['kernels_configuration']
    

    param_sizes = [net.input_dim[0] * width_configuration[0] * (kernels_configuration[0] ** 2) * 2]
    flop_counts = [flops_per_block(input_dim=net.input_dim[1], kernel_size=kernels_configuration[0], 
                                  in_channel=net.input_dim[0], out_channel=width_configuration[0])]

    input_dim = net.input_dim[1] // 2
    for i in range(1, len(kernels_configuration)):

        param_sizes += [(kernels_configuration[i] ** 2) * width_configuration[i - 1] * width_configuration[i] 
                        + (kernels_configuration[i] ** 2 * width_configuration[i] ** 2)]

        flop_counts += [flops_per_block(input_dim=input_dim, kernel_size=kernels_configuration[i], 
                                  in_channel=width_configuration[i - 1], out_channel=width_configuration[i])]
        if i % net.max_depth == 0:
            input_dim //= 2

    
    total_param_size = param_sizes[0]
    total_flops = flop_counts[0]
    i = 0
    for d in depth_configuration:
        total_param_size += sum(param_sizes[i * net.max_depth + 1:i * net.max_depth + d + 1])
        total_flops += sum(flop_counts[i * net.max_depth + 1:i * net.max_depth + d + 1])
        i += 1
        

    return total_param_size / 1000/ 1000, total_flops / 1000 / 1000
    
def count_inverted_residual_param(input_dim, in_channel, out_channel, kernel_size, expansion_ratio):
    hidden_out_channel = in_channel * expansion_ratio
    
    flops = flops_per_block(input_dim, 1, in_channel, hidden_out_channel, False)
    param_size = in_channel * hidden_out_channel * 1 * 1
    
    flops += flops_per_block(input_dim, kernel_size, 1, hidden_out_channel, False)
    param_size += hidden_out_channel * 1 * kernel_size * kernel_size
    
    flops += flops_per_block(input_dim, 1, hidden_out_channel, out_channel, False)
    param_size += hidden_out_channel * out_channel * 1 * 1
    
    return param_size, flops


def calculate_net_size_from_mobilenet_config(net):
    net_configuration = copy.deepcopy(net.get_configuration())
    
    depth_configuration = net_configuration['depth_configuration']
    width_expansion_configuration = net_configuration['width_expansion_configuration']
    width_configuration = net_configuration['width_configuration']
    width_configuration.insert(0, net.initial_out_channel)

    
    kernels_configuration = net_configuration['kernels_configuration']
    

    params, flops = count_inverted_residual_param(net.input_resolution, net.input_dim[0], width_configuration[0],
                                               kernels_configuration[0], width_expansion_configuration[0]) 
    param_sizes = [params]
    flops_count = [flops]


    input_dim = net.input_resolution // 2
    for i in range(1, len(kernels_configuration)):
        params, flops = count_inverted_residual_param(input_dim, width_configuration[i - 1], width_configuration[i],
                                               kernels_configuration[i], width_expansion_configuration[i])
        param_sizes += [params]
        flops_count += [flops]
        

        if i % net.max_depth == 0:
            input_dim //= 2

    total_param_size = param_sizes[0]
    total_flops = flops_count[0]
    i = 0
    for d in depth_configuration:
        total_param_size += sum(param_sizes[i * net.max_depth + 1:i * net.max_depth + d + 1])
        total_flops += sum(flops_count[i * net.max_depth + 1:i * net.max_depth + d + 1])

        i += 1
        
    return total_param_size / 1000/ 1000, total_flops / 1000 / 1000

# Modified from https://stats.stackexchange.com/questions/291843/how-to-understand-calculate-flops-of-the-neural-network-model
def flops_per_block(input_dim, kernel_size, in_channel, out_channel, repeat=True):
    mac = kernel_size * kernel_size * in_channel * input_dim * input_dim * out_channel
    
    input_shape = (in_channel,input_dim,input_dim) # Format:(channels, rows,cols)
    conv_filter = (out_channel,in_channel,kernel_size,kernel_size)  # Format: (num_filters, channels, rows, cols)
    stride = 1
    padding = kernel_size//2
    activation = 'relu'

    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    flops_per_instance = n    # general defination for number of flops (n: multiplications and n-1: additions)

    num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2*padding) / stride ) + 1  # for rows
    num_instances_per_filter *= (( input_shape[2] - conv_filter[3] + 2*padding) / stride ) + 1 # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]    # multiply with number of filters

    if activation == 'relu':
        # Here one can add number of flops required
        # Relu takes 1 comparison and 1 multiplication
        # Assuming for Relu: number of flops equal to length of input vector
        total_flops_per_layer += conv_filter[0]*num_instances_per_filter

    if repeat:
        total_flops_per_layer += flops_per_block(input_dim=input_dim, kernel_size=kernel_size, 
                        in_channel=out_channel, out_channel=out_channel, repeat=False)
        
    return total_flops_per_layer

def validate(net, val_generator, score_fn, loss_fn, device):
    # calibrate_batchnorm(net, val_generator)
    val_generator.dataset.set_input_dim(randomize=True, permute_from=[net.input_resolution])
    with torch.no_grad():
        net.eval()
        i = 0
        all_score = []
        all_losses = []
        for image, label in val_generator:
            
            image, label = image.to(device), label.to(device)

            if net.distill_knowledge:
                output, _ = net(image)
            else:
                output = net(image)
            

            # Validation loss
            loss = loss_fn(output, label).detach().cpu().numpy()
            all_losses += [loss]
            # Validation accuracy
            output = output.cpu().numpy()
            label = label.cpu().numpy()
            all_score += [score_fn(output, label)]

            i += 1
        
    net.train()
    return np.mean(all_score), np.mean(all_losses)