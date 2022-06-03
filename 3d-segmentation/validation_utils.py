import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from elastic_3d_unet import ConvBlock

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
        
        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)  # 1, C, 1, 1
        batch_var = (x - batch_mean) * (x - batch_mean)
        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)

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
def calibrate_batchnorm(net, generator, data=None, device_type='cuda'):

    device = torch.device("cuda" if device_type == 'cuda' else "cpu")

    bn_mean = {}
    bn_var = {}
    # print(net.layers_dict['batchNorm_sec_1_conv_4'].running_mean)
    generator.dataset.set_input_dim(randomize=True, permute_from=[net.input_resolution])
    copy_net = copy.deepcopy(net)

    for block_name, block in copy_net.named_modules():
        if not isinstance(block, ConvBlock):
            continue
        block.bn_calibration_mode = True
        for name, m in block.named_modules():
            if isinstance(m, nn.BatchNorm3d):
                name = block_name + '-' + name
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()
                m.forward = new_forward(m, bn_mean[name], bn_var[name])
                
    copy_net.train()
    # print('Calibrating BatchNorm mean and variance...')
    with torch.no_grad():
        i = 0
        if data != None:
            copy_net(data)
        else:
            for volume, _ in generator:
                volume = volume.to(device)
                copy_net(volume)
                i += 1

    del copy_net
    for block_name, block in net.named_modules():

        if not isinstance(block, ConvBlock):
            continue
        for name, m in block.named_modules():
            if isinstance(m, nn.BatchNorm3d):
                name = block_name + '-' + name
                # print('Calibrating {}-{}...'.format(block_name, name))
                if isinstance(bn_mean[name].avg, int):
                    continue

                feature_dim = bn_mean[name].avg.size(0)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

def dice_iou_combined(output, mask, epsilon=1e-5):
    intersection = (output * mask).sum()
    union = output.sum() + mask.sum()
    dice = (2 * intersection + epsilon)/ (union + epsilon)
    iou = (intersection + epsilon)/ (union - intersection + epsilon)
    return dice, iou

def dice_iou_categorical_combined(output, mask, epsilon=1e-5, axes=3):
    intersection = sum_on_dimension(output * mask, axes)
    dice = (2.*intersection + epsilon)/(sum_on_dimension(output + mask, axes) + epsilon)
    iou = (intersection + epsilon)/((sum_on_dimension(output + mask, axes) - intersection) + epsilon)
    return dice, iou

def dice_score(output, mask, epsilon=1e-5):
    intersection = (output * mask).sum()
    union = output.sum() + mask.sum()
    score = (2 * intersection + epsilon)/ (union + epsilon)
    return score

def dice_score_categorical(output, mask, epsilon=1e-5, axes=3):
    intersection = sum_on_dimension(output * mask, axes)
    score = (2.*intersection + epsilon)/(sum_on_dimension(output + mask, axes) + epsilon)
    return score

def iou_score(output, mask, epsilon=1e-5):
    intersection = (output * mask).sum()
    union = output.sum() + mask.sum()
    score = (intersection + epsilon)/ (union - intersection + epsilon)
    return score

def iou_score_categorical(output, mask, epsilon=1e-5, axes=3):
    intersection = sum_on_dimension(output * mask, axes)
    score = (intersection + epsilon)/((sum_on_dimension(output + mask, axes) - intersection) + epsilon)
    return score

def combined_loss(output, mask, epsilon=1e-5):
    mse = F.mse_loss(output, mask)
    dice = dice_loss(output, mask)
    return torch.mean(mse + dice)

def sum_on_dimension(a, n, use_numpy=True):
    for _ in range(n):
        a = np.sum(a, axis=-1) if use_numpy else torch.sum(a, dim=-1)
    return a

def dice_loss_categorical(output, mask, epsilon=1e-5, axes=3):
    intersection = sum_on_dimension(output * mask, axes, use_numpy=False)
    dice = (2.*intersection + epsilon)/(sum_on_dimension(output + mask, axes, use_numpy=False) + epsilon)
    return torch.mean(torch.mean(torch.abs(1 - dice), axis=1))

def dice_loss(output, mask, epsilon=1e-5):
    intersection = (output * mask).sum()
    dice = (2.*intersection + epsilon)/(output.sum() + mask.sum() + epsilon)
    return torch.abs(1 - dice)

def log_cosh_dice_loss_categorical(output, mask, epsilon=1e-5):
    l_dice = dice_loss_categorical(output, mask, epsilon=epsilon)
    lcd_loss = torch.log(torch.cosh(l_dice))
    return lcd_loss

def log_cosh_dice_loss(output, mask, epsilon=1e-5):
    l_dice = dice_loss(output, mask, epsilon=epsilon)
    lcd_loss = torch.log(torch.cosh(l_dice))
    return lcd_loss


def feature_layer_loss(output, mask, epsilon=1e-5):
    total_loss = 0
    for i in range(len(output)):
        out_channel = output[i].shape[1]
        mask_channel = mask[i].shape[1]
        n_channel = min([out_channel, mask_channel])
        resize_dim = [mask[i].size()[-1]]*3
        output[i][:,:n_channel] = F.interpolate(output[i][:,:n_channel], size=resize_dim)
        total_loss += F.mse_loss(output[i][:,:n_channel], mask[i][:,:n_channel])
    return total_loss / len(output)

def output_layer_loss(output, teacher_output):
    return F.mse_loss(output, teacher_output)

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

def calculate_net_size_from_3d_unet_config(net):
    net_configuration = copy.deepcopy(net.get_configuration())

    depth_configuration = net_configuration['depth_configuration']
    width_configuration = net_configuration['width_configuration']
    width_configuration.insert(-1, net.output_n)

    
    kernels_configuration = net_configuration['kernels_configuration']
    

    param_sizes = []
    flop_counts = []

    input_dim = net.input_resolution
    for i in range(0, len(kernels_configuration) - 1):
        
        if (i+1) % net.max_depth == 1 and i >= (net.max_depth * (net.max_sections // 2) + net.max_depth):
            input_dim *= 2
        
        param_sizes += [(kernels_configuration[i] ** 2) * width_configuration[i - 1] * width_configuration[i] 
                        + (kernels_configuration[i] ** 2 * width_configuration[i] ** 2)]

        flop_counts += [flops_per_block(input_dim=input_dim, kernel_size=kernels_configuration[i], 
                                  in_channel=width_configuration[i - 1], out_channel=width_configuration[i],
                                       repeat=False)]
        if (i+1) % net.max_depth == 0 and i < (net.max_depth * net.max_sections // 2):
            input_dim //= 2
        
        
    
    param_sizes += [net.input_resolution * width_configuration[-1] * (kernels_configuration[-1] ** 2) * 2]
    flop_counts += [flops_per_block(input_dim=net.input_resolution, kernel_size=kernels_configuration[-1], 
                                  in_channel=net.width_configuration[-2], out_channel=width_configuration[-1],
                                   repeat=False)]

    
    total_param_size = 0
    total_flops = 0
    i = 0
    for d in depth_configuration:
        total_param_size += sum(param_sizes[i * net.max_depth + 1:i * net.max_depth + d + 1])
        total_flops += sum(flop_counts[i * net.max_depth + 1:i * net.max_depth + d + 1])
        i += 1
    
    total_param_size += param_sizes[-1]
    total_flops += flop_counts[-1]

    return total_param_size / 1000/ 1000, total_flops / 1000 / 1000


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
            
            resize_dim = [label.size()[-1]]*3
            output = F.interpolate(output, size=resize_dim)

            # Validation loss
            loss = loss_fn(output, label).detach().cpu().numpy()
            all_losses += [loss]

            i += 1
        
    net.train()
    return np.mean(all_score), np.mean(all_losses)