import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np


class Classifier(nn.Module):
    def __init__(self, flatten_input, output_n):
        super(Classifier, self).__init__()
        self.layers_dict = {}
        self.layers_dict['linear_1'] = nn.Linear(flatten_input, 64)
        self.layers_dict['linear_2'] = nn.Linear(64, output_n)
        self.layers_dict = nn.ModuleDict(self.layers_dict)
        self.sampling_mode = False
        self.feature_size = 0
        self.active_params = 0
        self.flatten_input = flatten_input
    
    def enable_sampling_mode(self):
        self.sampling_mode = True
    
    def disable_sampling_mode(self):
        self.feature_size = 0
        self.active_params = 0
        self.sampling_mode = False
    
    def extract_weights_biases(self, layer_name, in_features):
        weights = self.layers_dict[layer_name].weight[:, :in_features]
        biases = self.layers_dict[layer_name].bias
        return weights, biases

    def sort_all_weights(self):
        for i in range(2):
            self.sort_weights('linear_{}'.format(i + 1))

    def sort_weights(self, conv_layer_name):
        importance = torch.sum(torch.abs(self.layers_dict[conv_layer_name].weight.data), dim=(1))

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0)

        self.layers_dict[conv_layer_name].weight.data = torch.index_select(self.layers_dict[conv_layer_name].weight.data, 0, sorted_idx)
        self.layers_dict[conv_layer_name].bias.data = torch.index_select(self.layers_dict[conv_layer_name].bias.data, 0, sorted_idx)

    def calculate_param_size(self):
        return self.active_params #*32 /8 /1024 /1024 # MB

    def calculate_feature_size(self):
        return self.feature_size #* 32 / 8 / 1024 / 1024 # MB
    
    def freeze_block(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_block(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = F.avg_pool2d(x, x.size(-1))
        x = torch.flatten(x, start_dim=1)
        if self.sampling_mode:
            self.feature_size += np.prod(x.size())
        
        weights, biases = self.extract_weights_biases('linear_1', x.size(1))
        weights, biases = weights.contiguous(), biases.contiguous()

        if self.sampling_mode:
            self.active_params += (np.prod(weights.size()) + np.prod(biases.size()))
            x = x.matmul(weights.t()) + biases
        else:
            x = F.linear(x, weights, biases)

        x = torch.relu(x)
        if self.sampling_mode:
            self.feature_size += np.prod(x.size())

        weights, biases = self.extract_weights_biases('linear_2', x.size(1))
        if self.sampling_mode:
            self.active_params += (np.prod(weights.size()) + np.prod(biases.size()))
            x = x.matmul(weights.t()) + biases
        else:
            x = F.linear(x, weights, biases)
    
        
        if self.sampling_mode:
            self.feature_size += np.prod(x.size())
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio=[3,4,6], 
                 kernels=[3,5,7], expanding_kernels=False, keep_channel_order=False, bn_track_running_stats=True):
        super(InvertedResidual, self).__init__()

        self.kernels = kernels
        self.kernels.sort(reverse=True)
        self.max_kernel = max(kernels)
        self.min_kernel = min(kernels)

        self.current_kernel = min(kernels) if expanding_kernels else max(kernels)
        self.expand_ratio = expand_ratio
        self.current_expand_ratio = max(expand_ratio)
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.max_out_channel = out_channel
        
        self.expanding_kernels = expanding_kernels
        self.keep_channel_order = keep_channel_order
        
        self.layers_dict = {}

        self.hidden_out = in_channel * self.current_expand_ratio
        self.max_hidden_out = self.hidden_out
        
        self.sorted_idx = {
            'hidden_channel': torch.arange(self.hidden_out).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
            'out_channel': torch.arange(out_channel).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))}

        layers = []
        
        
        self.layers_dict['expand_conv'] = nn.Conv2d(in_channels=in_channel, out_channels=self.hidden_out, 
                                                        kernel_size=1, bias=False)
        if expanding_kernels:
            
            for j in range(self.min_kernel, self.max_kernel + 1, 2):
                self.layers_dict['depthwise_conv_{}x{}'.format(j, j)] = nn.Conv2d(self.hidden_out, self.hidden_out, 
                                                                                  j, bias=False, groups=self.hidden_out)
        else:
            
            self.layers_dict['depthwise_conv'] = nn.Conv2d(in_channels=self.hidden_out, out_channels=self.hidden_out, 
                                                           kernel_size=self.max_kernel, bias=False, groups=self.hidden_out)
            
            self.layers_dict['depthwise_conv_transform_matrix'] = nn.ParameterList([nn.Parameter(
                torch.rand([k**2, k**2])) for k in range(self.max_kernel - 2, self.min_kernel - 1, -2)])

            for j in range(self.max_kernel - 2, self.min_kernel - 1, -2):
                self.layers_dict['depthwise_conv_{}x{}'.format(j, j)] = nn.Conv2d(self.hidden_out, self.hidden_out, 
                                                                                  j, bias=False, groups=self.hidden_out)

        
        self.layers_dict['pointwise_conv'] = nn.Conv2d(in_channels=self.hidden_out, out_channels=out_channel, 
                                                           kernel_size=1, bias=False)
        

        self.layers_dict['batchnorm'] = nn.BatchNorm2d(out_channel, track_running_stats=bn_track_running_stats)

        self.layers_dict = nn.ModuleDict(self.layers_dict)
        
        self.feature_size = 0
        self.active_params = 0
        self.sampling_mode = False
        self.bn_track_running_stats = bn_track_running_stats
        self.bn_calibration_mode = False
        
    
    def enable_sampling_mode(self):
        self.sampling_mode = True
    
    def disable_sampling_mode(self):
        self.feature_size = 0
        self.active_params = 0
        self.sampling_mode = False

    def set_current_kernel(self, k):
        self.current_kernel = k
        if k > self.min_kernel and self.expanding_kernels:
            self.reset_expanding_kernel_weights()

    def reset_expanding_kernel_weights(self):
        for k in range(self.min_kernel + 2, self.max_kernel + 1, 2):
            current_k_weights = self.layers_dict['depthwise_conv_{}x{}'.format(k, k)].weight[:, :, :, :].contiguous()
            smaller_k_weights = self.layers_dict['depthwise_conv_{}x{}'.format(k - 2, k - 2)].weight[:, :, :, :]
            self.layers_dict['depthwise_conv_{}x{}'.format(k, k)].weight.data[:, :, 1:k-1, 1:k-1] = smaller_k_weights.data
    
    def set_in_out_channel(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_out = self.in_channel * self.current_expand_ratio
        if self.keep_channel_order:
            conv_layer_name = 'expand_conv'.format(self.current_kernel, self.current_kernel)
            self.sorted_idx['hidden_channel'] = self.get_sorted_weights_index(conv_layer_name=conv_layer_name)
            
            conv_layer_name = 'pointwise_conv'.format(self.current_kernel, self.current_kernel)
            self.sorted_idx['out_channel'] = self.get_sorted_weights_index(conv_layer_name=conv_layer_name)

            # reorganize batchnorm layer
            sorted_idx = self.sorted_idx['out_channel']
            select_idx, _ = torch.sort(sorted_idx, dim=0)

            self.layers_dict['batchnorm'].weight.data = torch.index_select(self.layers_dict['batchnorm'].weight.data, 0, select_idx)
            self.layers_dict['batchnorm'].bias.data = torch.index_select(self.layers_dict['batchnorm'].bias.data, 0, select_idx)
            if self.bn_track_running_stats:
                self.layers_dict['batchnorm'].running_mean.data = torch.index_select(self.layers_dict['batchnorm'].running_mean.data, 0, select_idx)
                self.layers_dict['batchnorm'].running_var.data = torch.index_select(self.layers_dict['batchnorm'].running_var.data, 0, select_idx)
    
    def get_sorted_weights_index(self, conv_layer_name=None, weight=None):
        if conv_layer_name != None:
            importance = torch.sum(torch.abs(self.layers_dict[conv_layer_name].weight.data), dim=(1,2,3))
        else:
            importance = torch.sum(torch.abs(weight.data), dim=(1,2,3))
        _, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx
    
    def generate_channels(self, out_channel, layer_type, conv_layer_name=None, weight=None):
        sorted_idx = self.sorted_idx[layer_type][:out_channel]
        select_idx, _ = torch.sort(sorted_idx, dim=0)

        select_idx = select_idx.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        


        if conv_layer_name != None:
            if self.sampling_mode:
                new_weights = self.layers_dict[conv_layer_name].weight[:out_channel]
            else:
                new_weights = torch.index_select(self.layers_dict[conv_layer_name].weight, 0, select_idx)
        else:
            if self.sampling_mode:
                new_weights = weight[:out_channel]
            else:
                new_weights = torch.index_select(weight, 0, select_idx)
        return new_weights
    
    def generate_expanding_kernels(self, k, in_channel, out_channel):
        return self.layers_dict['depthwise_conv_{}x{}'.format(k, k)].weight[:out_channel, :in_channel, :, :]
    
    def set_width_expansion_ratio(self, current_expand_ratio):
        self.current_expand_ratio = current_expand_ratio
        self.hidden_out = self.in_channel * self.current_expand_ratio
        
    def clear_middle_kernel_grad(self):
        k = self.current_kernel
        if k == self.min_kernel:
            return
        if self.layers_dict['depthwise_conv_{}x{}'.format(k, k)].weight.grad != None:
            self.layers_dict['depthwise_conv_{}x{}'.format(k, k)].weight.grad[:, :, 1:k-1, 1:k-1] = 0.

    def sort_all_channels(self, conv_type=''):
        if self.expanding_kernels:
            conv_type = conv_type + '_{}x{}'.format(self.current_kernel, self.current_kernel)
        self.sort_channel(conv_type)
        

    def sort_channel(self, conv_layer_name, shrink_stage=0):
        importance = torch.sum(torch.abs(self.layers_dict[conv_layer_name].weight.data), dim=(1,2,3))

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0)

        self.layers_dict[conv_layer_name].weight.data = torch.index_select(self.layers_dict[conv_layer_name].weight.data, 0, sorted_idx)

        # Batch Normalization
        if 'pointwise' in conv_layer_name:
            self.layers_dict['batchnorm'].weight.data = torch.index_select(self.layers_dict['batchnorm'].weight.data, 0, sorted_idx)
            self.layers_dict['batchnorm'].bias.data = torch.index_select(self.layers_dict['batchnorm'].bias.data, 0, sorted_idx)
            if self.bn_track_running_stats:
                self.layers_dict['batchnorm'].running_mean.data = torch.index_select(self.layers_dict['batchnorm'].running_mean.data, 0, sorted_idx)
                self.layers_dict['batchnorm'].running_var.data = torch.index_select(self.layers_dict['batchnorm'].running_var.data, 0, sorted_idx)

    def generate_kernels(self, k, in_channel, out_channel, conv_layer_name='depthwise_conv'):
        if conv_layer_name != 'depthwise_conv':
            return self.layers_dict[conv_layer_name].weight[:out_channel, :in_channel, :, :]
            
            
        trans_mat_name = '{}_transform_matrix'.format(conv_layer_name)

        if k == self.max_kernel:
            return self.layers_dict[conv_layer_name].weight[:out_channel, :in_channel, :, :]
            
        if self.sampling_mode:
            return self.layers_dict[conv_layer_name + '_{}x{}'.format(k, k)].weight.data[:out_channel, :in_channel, :, :]
            

        kernel_dict_ind = 0
        current_k_weights = self.layers_dict[conv_layer_name].weight.data[:out_channel, :in_channel, :, :]
        
        for current_k in self.kernels[1:]:
            if current_k < k:              
                break

            in_channel, out_channel = current_k_weights.size(1), current_k_weights.size(0)
            current_k_weights = current_k_weights[:, :, 1:current_k+1, 1:current_k+1]
            current_k_weights = current_k_weights.contiguous()
            current_k_weights = current_k_weights.view(current_k_weights.size(0), current_k_weights.size(1), -1)
            current_k_weights = current_k_weights.view(-1, current_k_weights.size(2))

            current_k_weights = F.linear(current_k_weights, self.layers_dict[trans_mat_name][kernel_dict_ind])
            
            current_k_weights = current_k_weights.view(out_channel, in_channel, current_k, current_k)
            kernel_dict_ind += 1
        
        self.layers_dict[conv_layer_name + '_{}x{}'.format(k, k)].weight.data[:out_channel, :in_channel, :, :] = current_k_weights
        
        return current_k_weights

    def batch_norm_func(self, x, bn, feature_dim):
        if bn.num_features == feature_dim or self.bn_calibration_mode:
            return bn(x)
        else:
            exponential_average_factor = 0.0
            
            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  
                        exponential_average_factor = bn.momentum
            
            w = bn.weight
            b = bn.bias
            running_mean = bn.running_mean if bn.track_running_stats else None
            running_var = bn.running_var if bn.track_running_stats else None

            if not bn.track_running_stats:
                return F.batch_norm(
                x, running_mean, running_var, w[:feature_dim],
                b[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,)

            return F.batch_norm(
                x, running_mean[:feature_dim], running_var[:feature_dim], w[:feature_dim],
                b[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,)
    
    def freeze_block(self):
        for param in self.parameters():
            param.requires_grad = False

        self.layers_dict['batchnorm'].eval()

    def unfreeze_block(self):
        for param in self.parameters():
            param.requires_grad = True

        self.layers_dict['batchnorm'].train()
        
    def calculate_param_size(self):
        return self.active_params #*32 /8 /1024 /1024 # MB

    def calculate_feature_size(self):
        return self.feature_size #* 32 / 8 / 1024 / 1024 # MB
    
    def forward(self, x):
        
        kernels = self.generate_kernels(1, x.size(1), self.max_hidden_out, conv_layer_name='expand_conv')

        kernels = self.generate_channels(self.hidden_out, layer_type='hidden_channel', weight=kernels)

        
        if self.sampling_mode:
            self.active_params += np.prod(kernels.size())
        

        out = F.conv2d(x, kernels, padding=0)
        out = torch.relu(out)


        if self.sampling_mode:
            self.feature_size += np.prod(out.size())
        
        if self.expanding_kernels:
            kernels = self.generate_expanding_kernels(self.current_kernel, self.hidden_out, self.hidden_out)
        else:
            kernels = self.generate_kernels(self.current_kernel, self.hidden_out, self.hidden_out, conv_layer_name='depthwise_conv')
        

        if self.sampling_mode:
            self.active_params += np.prod(kernels.size())
        out = F.conv2d(out, kernels, padding=self.current_kernel//2, groups=out.size(1))
        
        out = torch.relu(out)
        
        if self.sampling_mode:
            self.feature_size += np.prod(out.size())
        
        kernels = self.generate_kernels(1, out.size(1), self.max_out_channel, conv_layer_name='pointwise_conv')

        kernels = self.generate_channels(self.out_channel, layer_type='out_channel', weight=kernels)
        
        
        if self.sampling_mode:
            self.active_params += np.prod(kernels.size())
        out = F.conv2d(out, kernels, padding=0)
        
        out = torch.relu(out)
        if self.sampling_mode:
            self.feature_size += np.prod(out.size())

        if out.size(1) != x.size(1):
            layers_diff = out.size(1) - x.size(1)
            pad = (0,0,0,0,layers_diff//2, layers_diff - layers_diff//2)
            x = F.pad(x, pad)

        out = out + x
        
        out = torch.relu(out)
        
        bn = self.layers_dict['batchnorm']
        out = self.batch_norm_func(out, bn, out.size(1))

        if self.sampling_mode:
            self.active_params += np.prod([8, out.size(1)])
            self.feature_size += np.prod(out.size())
        return out

class ElasticMobileNet(nn.Module):
    def __init__(self, input_dim=(4, 240, 240), kernels=[3,5,7], expansion_ratio=[3,4,6], distill_knowledge=False,
                 max_depth=3, output_n=4, sections=[2,3,4], expanding_kernels=False, keep_channel_order=False,
                 bn_track_running_stats=True):
        super(ElasticMobileNet, self).__init__()
        self.output_n=output_n
        self.sections=sections
        self.sections_configuration = max(self.sections)
        self.max_sections = max(self.sections)
        self.max_depth = max_depth
        self.expansion_ratio = expansion_ratio
        
        self.kernels = kernels
        self.kernels.sort(reverse=True)
        self.max_kernel = max(kernels)
        self.min_kernel = min(kernels)
        self.initial_out_channel = 16
        
        self.kernels_configuration=[self.max_kernel for i in range(self.max_depth * self.max_sections + 1)]
        self.first_kernel_configuration = self.kernels_configuration[0]
        self.depth_configuration = [self.max_depth for i in range(self.max_sections)]
        self.width_configuration = []
        self.width_expansion_configuration = [max(self.expansion_ratio) for _ in range(self.max_depth * self.max_sections + 1)]
        self.first_width_expansion_configuration = self.width_expansion_configuration[0]
        
        self.input_dim = input_dim
        self.input_resolution = input_dim[1]
        # Create dictionary of modules
        self.layers_dict = {}

        in_channel = input_dim[0]
        out_channel = self.initial_out_channel
        
        self.expanding_kernels = expanding_kernels
        self.keep_channel_order = keep_channel_order

        self.layers_dict['first_conv'] = InvertedResidual(in_channel=in_channel, out_channel=out_channel, 
                                                          expanding_kernels=expanding_kernels, keep_channel_order=keep_channel_order,
                                                          kernels=kernels, expand_ratio=expansion_ratio,
                                                          bn_track_running_stats=bn_track_running_stats)

        in_channel = out_channel
        out_channel = out_channel * 2

        for sec in range(self.max_sections):
            self.layers_dict['down_conv_{}'.format(sec)] = nn.MaxPool2d(2)
            for d in range(self.max_depth):
                self.width_configuration += [out_channel]
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)] = InvertedResidual(in_channel=in_channel, out_channel=out_channel,
                                                                                                    kernels=kernels, expand_ratio=expansion_ratio,
                                                                                                   expanding_kernels=expanding_kernels, keep_channel_order=keep_channel_order,
                                                                                                   bn_track_running_stats=bn_track_running_stats)
                in_channel = out_channel
            out_channel *= 2

        flatten_input = in_channel
        
        self.layers_dict['classifier'] = Classifier(flatten_input=flatten_input, output_n=self.output_n)

        self.layers_dict = nn.ModuleDict(self.layers_dict)

        self.sampling_mode = False
        self.active_params = 0
        self.feature_size = 0
        self.distill_knowledge= distill_knowledge
        self.best_configuration = None
    
    def sort_all_channels(self, conv_type=''):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, InvertedResidual): 
                layer.sort_all_channels(conv_type=conv_type)

        
    def enable_sampling_mode(self):
        self.sampling_mode = True
        self.layers_dict['first_conv'].enable_sampling_mode()
        for sec in range(self.max_sections):
            for d in range(self.max_depth):
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].enable_sampling_mode()
        self.layers_dict['classifier'].enable_sampling_mode()

    def disable_sampling_mode(self):
        self.sampling_mode = False
        self.layers_dict['first_conv'].disable_sampling_mode()
        for sec in range(self.max_sections):
            for d in range(self.max_depth):
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].disable_sampling_mode()
        self.layers_dict['classifier'].disable_sampling_mode()
        self.active_params = 0
        self.feature_size = 0

    def load_configuration(self, configuration):
        self.sections_configuration = configuration['sections_configuration']
        self.depth_configuration = configuration['depth_configuration']
        self.kernels_configuration = configuration['kernels_configuration']
        self.first_kernel_configuration = self.kernels_configuration[0]
        self.width_configuration = configuration['width_configuration']
        self.input_resolution = configuration['input_resolution']
        self.width_expansion_configuration = configuration['width_expansion_configuration']
        self.first_width_expansion_configuration = self.width_expansion_configuration[0]
    
    def get_configuration(self):
        return {
                'input_resolution': self.input_resolution,
                'kernels_configuration': self.kernels_configuration,
                'depth_configuration': self.depth_configuration,
                'width_configuration': self.width_configuration,
                'width_expansion_configuration': self.width_expansion_configuration,
                'sections_configuration':self.sections_configuration
                }
    
    # Not Supported yet
    def set_section_configuration(self, randomize=True, sections=None):
        self.sections_configuration = random.choice(self.sections) if randomize else sections

    def set_width_expansion_configuration(self, randomize=True, current_expansion_ratio=None, permute_from=None):
        if randomize:
            choices = self.expansion_ratio
            if permute_from is not None:
                choices = permute_from
            self.width_expansion_configuration = [random.choice(choices) for i in range(self.max_depth * self.max_sections + 1)]
        else:
            self.width_expansion_configuration = [current_expansion_ratio for i in range(self.max_depth * self.max_sections + 1)]
        self.first_width_expansion_configuration = self.width_expansion_configuration[0]

    def set_width_configuration(self, randomize=True, stage=0):
        self.width_configuration = []
        out_channel = self.initial_out_channel

        initial_power = int(math.log(out_channel, 2))
        
        factor = 1
        for sec in range(self.sections_configuration):
            choices = [2**(factor + initial_power - 1), 2**(factor + initial_power)]
            out_channel = random.choice(choices) if randomize else choices[stage]
            self.width_configuration += [out_channel for _ in range(self.max_depth)]
            factor += 1
    
    def set_kernels_configuration(self, randomize=True, kernel=None, permute_from=None):
        if randomize:
            choices = self.kernels
            if permute_from is not None:
                choices = permute_from
            self.kernels_configuration = [random.choice(choices) for i in range(self.max_depth * self.max_sections + 1)]
        else:
            self.kernels_configuration = [kernel for i in range(self.max_depth * self.max_sections + 1)]
        self.first_kernel_configuration = self.kernels_configuration[0]
        
        
    def set_depth_configuration(self, randomize=True, depth=4, freeze_output_layer=False, permute_from=None, current_depth_stage=None):
        if randomize:
            choices = [i for i in range(1, self.max_depth + 1)]
            if permute_from is not None:
                choices = permute_from
            self.depth_configuration = [random.choice(choices) for i in range(self.max_sections)]
        else:
            self.depth_configuration = [depth for i in range(self.max_sections)]

        if current_depth_stage:
            self.freeze_layers_by_stage(freeze_output_layer=False, current_depth_stage=current_depth_stage)
        else:
            self.freeze_layers(freeze_output_layer)
    
    def freeze_layers(self, freeze_output_layer=False):
        self.unfreeze_all_layers()
        for sec in range(self.sections_configuration):
            for d in range(self.depth_configuration[sec] - 1):
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].freeze_block()
            self.layers_dict['inverted_residual_block_{}_{}'.format(sec, self.depth_configuration[sec] - 1)].unfreeze_block()
        if freeze_output_layer:
            self.layers_dict['classifier'].freeze_block()

    def freeze_layers_by_stage(self, freeze_output_layer=False, current_depth_stage=0):
        self.unfreeze_all_layers()
        for sec in range(self.sections_configuration):
            for d in range(self.depth_configuration[sec]):
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].freeze_block()
            if self.depth_configuration[sec] < current_depth_stage + 1:
                continue
            self.layers_dict['inverted_residual_block_{}_{}'.format(sec, self.depth_configuration[sec] - 1)].unfreeze_block()
        if freeze_output_layer:
            self.layers_dict['classifier'].freeze_block()
        
    def unfreeze_all_layers(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, InvertedResidual):
                layer.unfreeze_block()
        self.layers_dict['classifier'].unfreeze_block()
    
    def freeze_all_middle_kernel_grads(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, InvertedResidual):
                layer.clear_middle_kernel_grad()
                
    def calculate_param_size(self):
        self.active_params += self.layers_dict['first_conv'].calculate_param_size()
        for sec in range(self.max_sections):
            for d in range(self.max_depth):
                self.active_params += self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].calculate_param_size()

        self.active_params += self.layers_dict['classifier'].calculate_param_size()
        return self.active_params 

    def calculate_feature_size(self):
        self.feature_size += self.layers_dict['first_conv'].calculate_feature_size()
        for sec in range(self.max_sections):
            for d in range(self.max_depth):
                self.feature_size += self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].calculate_feature_size()
        self.feature_size += self.layers_dict['classifier'].calculate_feature_size()
        return self.feature_size 
    
    def forward(self, x):
        layer_outputs = []
        self.layers_dict['first_conv'].set_current_kernel(self.kernels_configuration[0])
        self.layers_dict['first_conv'].set_in_out_channel(in_channel=x.size(1), out_channel=self.initial_out_channel)
        self.layers_dict['first_conv'].set_width_expansion_ratio(self.width_expansion_configuration[0])
        x = self.layers_dict['first_conv'](x)
        layer_outputs.append(x)

        for sec in range(self.sections_configuration):
            x = self.layers_dict['down_conv_{}'.format(sec)](x)
            for d in range(self.depth_configuration[sec]):
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].set_current_kernel(self.kernels_configuration[sec * self.max_depth + d + 1])
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].set_in_out_channel(in_channel=x.size(1), 
                                                                                                    out_channel=self.width_configuration[sec * self.max_depth + d])
                self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)].set_width_expansion_ratio(self.width_expansion_configuration[sec * self.max_depth + d + 1])
                x = self.layers_dict['inverted_residual_block_{}_{}'.format(sec, d)](x)
            layer_outputs.append(x)
        x = self.layers_dict['classifier'](x)
        if self.distill_knowledge:
            return x, layer_outputs
        return x
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print('Unmatched weight at {}: {}'.format(name, e))
