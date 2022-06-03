import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernels=[3,5], activation='relu', 
                 contains_batchnorm=True, expanding_kernels=False, keep_channel_order=True, bn_track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.kernels = kernels
        self.kernels.sort(reverse=True)
        self.max_kernel = max(kernels)
        self.min_kernel = min(kernels)
        self.current_kernel = min(kernels) if expanding_kernels else max(kernels)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.max_out_channel = out_channel

        self.activation = activation
        self.contains_batchnorm = contains_batchnorm
        self.sorted_idx = torch.arange(out_channel).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.layers_dict = {}
        if expanding_kernels:
            for j in range(self.min_kernel, self.max_kernel+1, 2):
                self.layers_dict['conv_{}x{}'.format(j, j)] = nn.Conv3d(in_channel, out_channel, j, bias=False)
        else:
            self.layers_dict['conv'] = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, 
                                                    kernel_size=self.max_kernel, bias=False)
            self.layers_dict['transform_matrix'] = nn.ParameterList([nn.Parameter(
                        torch.rand([k**3, k**3])) for k in range(self.max_kernel - 2, self.min_kernel - 1, -2)])
            
            for j in range(self.max_kernel - 2, self.min_kernel - 1, -2):
                self.layers_dict['conv_{}x{}'.format(j, j)] = nn.Conv3d(in_channel, out_channel, j, bias=False)

        self.layers_dict['batchnorm'] = nn.BatchNorm3d(out_channel, track_running_stats=bn_track_running_stats)

        self.layers_dict = nn.ModuleDict(self.layers_dict)
        self.expanding_kernels = expanding_kernels
        self.keep_channel_order = keep_channel_order
        self.sampling_mode = False
        self.bn_calibration_mode = False
        self.active_params = 0
        self.feature_size = 0

    def enable_sampling_mode(self):
        self.sampling_mode = True
    
    def disable_sampling_mode(self):
        self.sampling_mode = False
        self.active_params = 0
        self.feature_size = 0

    def set_current_kernel(self, k):
        self.current_kernel = k
        if k > self.min_kernel and self.expanding_kernels:
            self.reset_expanding_kernel_weights()

    def reset_expanding_kernel_weights(self):
        for k in range(self.min_kernel + 2, self.max_kernel + 1, 2):
            current_k_weights = self.layers_dict['conv_{}x{}'.format(k, k)].weight[:, :, :, :, :].contiguous()
            smaller_k_weights = self.layers_dict['conv_{}x{}'.format(k - 2, k - 2)].weight[:, :, :, :, :]
            self.layers_dict['conv_{}x{}'.format(k, k)].weight.data[:, :, 1:k-1, 1:k-1, 1:k-1] = smaller_k_weights.data

    
    def set_in_out_channel(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel

    def get_sorted_weights_index(self, conv_layer_name=None, weight=None):
        if conv_layer_name != None:
            importance = torch.sum(torch.abs(self.layers_dict[conv_layer_name].weight.data), dim=(1,2,3,4))
        else:
            importance = torch.sum(torch.abs(weight), dim=(1,2,3,4))
        _, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx

    def generate_channels(self, out_channel, conv_layer_name=None, weight=None):
        sorted_idx = self.sorted_idx[:out_channel]
        select_idx, _ = torch.sort(sorted_idx, dim=0)
        if conv_layer_name != None:
            new_weights = self.layers_dict[conv_layer_name].weight[select_idx]
        else:
            new_weights = weight[select_idx]
        
        return new_weights
        

    def generate_expanding_kernels(self, k, in_channel, out_channel):
        return self.layers_dict['conv_{}x{}'.format(k, k)].weight[:out_channel, :in_channel, :, :, :]
    
    def clear_middle_kernel_grad(self):
        k = self.current_kernel
        if k == self.min_kernel:
            return
        if self.layers_dict['conv_{}x{}'.format(k, k)].weight.grad != None:
            self.layers_dict['conv_{}x{}'.format(k, k)].weight.grad[:, :, 1:k-1, 1:k-1, 1:k-1] = 0.

    def generate_kernels(self, conv_layer_name, k, in_channel, out_channel):
        if k == self.max_kernel:
            return self.layers_dict[conv_layer_name].weight[:out_channel, :in_channel, :, :, :]
        if self.sampling_mode:
            return self.layers_dict[conv_layer_name + '_{}x{}'.format(k, k)].weight[:out_channel, :in_channel, :, :, :]

        kernel_dict_ind = 0
        current_k_weights = self.layers_dict[conv_layer_name].weight.data[:out_channel, :in_channel, :, :, :]
        
        for current_k in self.kernels[1:]:
            if current_k < k:              
                break

            in_channel, out_channel = current_k_weights.size(1), current_k_weights.size(0)
            current_k_weights = current_k_weights[:, :, 1:current_k+1, 1:current_k+1, 1:current_k+1]
            current_k_weights = current_k_weights.contiguous()
            current_k_weights = current_k_weights.view(current_k_weights.size(0), current_k_weights.size(1), -1)
            current_k_weights = current_k_weights.view(-1, current_k_weights.size(2))

            current_k_weights = F.linear(current_k_weights, self.layers_dict['transform_matrix'][kernel_dict_ind])

            current_k_weights = current_k_weights.view(out_channel, in_channel, current_k, current_k, current_k)
            kernel_dict_ind += 1
        
        self.layers_dict[conv_layer_name + '_{}x{}'.format(k, k)].weight.data[:out_channel, :in_channel, :, :, :] = current_k_weights
        return current_k_weights
    
    def sort_all_channel(self):

        for name, layer in self.layers_dict.items():
            if isinstance(layer, nn.Conv3d):
                if self.keep_channel_order:
                    self.sorted_idx = self.get_sorted_weights_index(conv_layer_name=name)
                    continue
                self.sort_channel(name)

    def sort_channel(self, conv_layer_name, shrink_stage=0):
        importance = torch.sum(torch.abs(self.layers_dict[conv_layer_name].weight.data), dim=(1,2,3,4))

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0)
        select_idx = original_idx[sorted_idx]
        
        self.layers_dict[conv_layer_name].weight.data = torch.index_select(self.layers_dict[conv_layer_name].weight.data, 0, sorted_idx)

        # Batch Normalization
        
        self.layers_dict['batchnorm'].weight.data = torch.index_select(self.layers_dict['batchnorm'].weight.data, 0, sorted_idx)
        self.layers_dict['batchnorm'].bias.data = torch.index_select(self.layers_dict['batchnorm'].bias.data, 0, sorted_idx)
        if self.layers_dict['batchnorm'].track_running_stats:
            self.layers_dict['batchnorm'].running_mean.data = torch.index_select(self.layers_dict['batchnorm'].running_mean.data, 0, sorted_idx)
            self.layers_dict['batchnorm'].running_var.data = torch.index_select(self.layers_dict['batchnorm'].running_var.data, 0, sorted_idx)
        
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
            
            sorted_idx = self.sorted_idx[:feature_dim]
            select_idx, _ = torch.sort(sorted_idx, dim=0)

            w = bn.weight[select_idx] 
            b = bn.bias[select_idx]
            running_mean = bn.running_mean[select_idx] if bn.track_running_stats else None
            running_var = bn.running_var[select_idx] if bn.track_running_stats else None
            
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

    def forward(self, x):
        if self.expanding_kernels:
            kernels = self.generate_expanding_kernels(self.current_kernel, self.in_channel, self.max_out_channel)
        else:
            kernels = self.generate_kernels('conv', self.current_kernel, self.in_channel, self.max_out_channel)
        
        kernels = self.generate_channels(out_channel=self.out_channel, weight=kernels)


        if self.sampling_mode:
            self.active_params += np.prod(kernels.size())
        x = F.conv3d(x, kernels, padding=self.current_kernel//2)
        if self.activation == 'relu':
            x = torch.relu(x)
        else:
            x = relu1(x)
        if self.contains_batchnorm:
            bn = self.layers_dict['batchnorm']
            x = self.batch_norm_func(x, bn, self.out_channel)
            if self.sampling_mode:
                self.active_params += np.prod([4, self.out_channel])
        if self.sampling_mode:
            self.feature_size += np.prod(x.size())
        return x

    
def relu1(x):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.minimum(torch.maximum(torch.tensor(0).to(device),x), torch.tensor(1).to(device))

class Elastic3DUNet(nn.Module):
    """docstring for Elastic3DUNet"""
    def __init__(self, kernels=[3,5], sections=[5, 7, 9], max_depth=4, output_n=4, 
                 input_dim=(4, 128, 128, 128), distill_knowledge=False, initial_out_channel=8,
                 expanding_kernels=False, train_only_current_stage=False, keep_channel_order=True,bn_track_running_stats=True):
        super(Elastic3DUNet, self).__init__()
        self.output_n=output_n
        self.sections=sections
        self.sections_configuration = max(self.sections)
        self.skip_sections = []
        self.max_sections = max(self.sections)
        self.input_resolution = input_dim[1]
        self.max_depth = max_depth

        self.kernels = kernels
        self.kernels.sort(reverse=True)
        self.max_kernel = max(kernels)
        self.min_kernel = min(kernels)
        if expanding_kernels:
            self.kernels_configuration=[self.min_kernel for i in range(self.max_depth * self.sections_configuration + 1)]
        else:
            self.kernels_configuration=[self.max_kernel for i in range(self.max_depth * self.sections_configuration + 1)]

        self.down_sample_kernels_configuration=[2 for i in range(self.sections_configuration // 2)]
        self.last_kernel_configuration = self.kernels_configuration[-1]

        self.depth_configuration = [self.max_depth for i in range(self.sections_configuration)]
        
        self.width_configuration = []

        # Create dictionary of modules
        self.layers_dict = {}

        in_channel = input_dim[0]
        self.initial_out_channel = initial_out_channel
        out_channel = self.initial_out_channel
        for section in range(self.sections_configuration):
            if section > self.sections_configuration // 2:
                self.layers_dict['conv3D_up_{}'.format(self.sections_configuration - section)] = nn.Upsample(scale_factor=2)

            for i in range(self.max_depth):
                if i == 0:
                    if section <= self.sections_configuration // 2:
                        out_channel *= 2
                    else:
                        in_channel, out_channel = in_channel + out_channel//2, in_channel//2

                self.width_configuration.append(out_channel)

                self.layers_dict['conv3D_sec_{}_conv_{}'.format(section + 1, i + 1)] = ConvBlock(in_channel=in_channel, out_channel=out_channel, 
                                                                                                 kernels=self.kernels, expanding_kernels=expanding_kernels, 
                                                                                                 keep_channel_order=keep_channel_order, 
                                                                                                 bn_track_running_stats=bn_track_running_stats)

                in_channel = out_channel
                

            if section < self.sections_configuration // 2:
                self.layers_dict['conv3D_down_{}'.format(section + 1)] = nn.MaxPool3d(2)
                

        out_channel = self.output_n
        self.width_configuration.append(out_channel)

        self.layers_dict['conv3D_final_output'] = ConvBlock(in_channel=in_channel, out_channel=out_channel, 
                                                            kernels=self.kernels, activation='sigmoid', contains_batchnorm=False, 
                                                            expanding_kernels=expanding_kernels, keep_channel_order=keep_channel_order, 
                                                            bn_track_running_stats=bn_track_running_stats)
        
        self.layers_dict = nn.ModuleDict(self.layers_dict)

        self.sampling_mode = False
        self.active_params = 0
        self.feature_size = 0

        self.distill_knowledge = distill_knowledge
        self.expanding_kernels = expanding_kernels
        self.train_only_current_stage = train_only_current_stage
        self.keep_channel_order=keep_channel_order
        self.best_configuration = None

    def enable_sampling_mode(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                layer.enable_sampling_mode()
        self.sampling_mode = True
    
    def disable_sampling_mode(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                layer.disable_sampling_mode()
        self.sampling_mode = False
        self.active_params = 0
        self.feature_size = 0

    def sort_all_channels(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                layer.sort_all_channel()

    def set_section_configuration(self, randomize=True, section=9):
        self.sections_configuration = random.choice(self.sections) if randomize else section
        sections_pruned = (self.max_sections - self.sections_configuration) // 2
        mid_section = self.max_sections // 2
        self.skip_sections = [mid_section - i for i in range(1, sections_pruned + 1)] + [mid_section + i for i in range(1, sections_pruned + 1)]
        self.width_configuration = []
        factor = 0
        
        factor = -1
        for i in range(mid_section + 1):
            if i not in self.skip_sections:
                factor += 1
            self.width_configuration.append([2**(factor + 4)] * self.max_depth)
        
        for i in range(mid_section - 1, -1, -1):
            if i not in self.skip_sections:
                    factor -= 1  
            self.width_configuration.append([2**(factor + 4)] * self.max_depth)
        
        self.width_configuration = [k for k_list in self.width_configuration for k in k_list]
        self.width_configuration.append(self.output_n)

    def set_kernels_configuration(self, randomize=True, kernel=5, permute_from=None):
        if randomize:
            choices = self.kernels
            if permute_from is not None:
                choices = permute_from
            self.kernels_configuration = [random.choice(choices) for i in range(self.max_depth * self.max_sections + 1)]
        else:
            self.kernels_configuration = [kernel for i in range(self.max_depth * self.max_sections + 1)]
        self.last_kernel_configuration = self.kernels_configuration[-1]

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
                self.layers_dict['conv3D_sec_{}_conv_{}'.format(sec + 1, d + 1)].freeze_block()
            self.layers_dict['conv3D_sec_{}_conv_{}'.format(sec + 1, self.depth_configuration[sec])].unfreeze_block()
        if freeze_output_layer:
            self.layers_dict['conv3D_final_output'].freeze_block()

    def freeze_layers_by_stage(self, freeze_output_layer=False, current_depth_stage=0):
        self.unfreeze_all_layers()
        for sec in range(self.sections_configuration):
            for d in range(self.depth_configuration[sec]):
                self.layers_dict['conv3D_sec_{}_conv_{}'.format(sec + 1, d + 1)].freeze_block()
            if self.depth_configuration[sec] < current_depth_stage + 1:
                continue
            self.layers_dict['conv3D_sec_{}_conv_{}'.format(sec + 1, self.depth_configuration[sec])].unfreeze_block()
        if freeze_output_layer:
            self.layers_dict['conv3D_final_output'].freeze_block()

    def unfreeze_all_layers(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                layer.unfreeze_block()

    def freeze_all_middle_kernel_grads(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                layer.clear_middle_kernel_grad()

    def set_width_configuration(self, randomize=True, stage=0):
        sections_pruned = (self.max_sections - self.sections_configuration) // 2
        mid_section = self.max_sections // 2
        self.skip_sections = [mid_section - i for i in range(1, sections_pruned + 1)] + [mid_section + i for i in range(1, sections_pruned + 1)]
        self.width_configuration = []
        factor = 0
        initial_power = int(math.log(self.initial_out_channel, 2))
        for i in range(mid_section + 1):
            if i not in self.skip_sections:
                factor += 1
            if randomize:
                self.width_configuration.append([random.choice([2**(factor + initial_power - 1), 2**(factor + initial_power)])] * self.max_depth)
            else:
                self.width_configuration.append([[2**(factor + initial_power - 1), 2**(factor + initial_power)][stage]] * self.max_depth)
        for i in range(mid_section - 1, -1, -1):
            if i not in self.skip_sections:
                    factor -= 1  
            if randomize:
                self.width_configuration.append([random.choice([2**(factor + initial_power - 1), 2**(factor + initial_power)])] * self.max_depth)
            else:
                self.width_configuration.append([[2**(factor + initial_power - 1), 2**(factor + initial_power)][stage]] * self.max_depth)
        
        self.width_configuration = [k for k_list in self.width_configuration for k in k_list]
        self.width_configuration.append(self.output_n)

    def load_configuration(self, configuration):
        self.sections_configuration = configuration['sections_configuration']
        self.depth_configuration = configuration['depth_configuration']
        self.kernels_configuration = configuration['kernels_configuration']
        self.width_configuration = configuration['width_configuration']
        self.input_resolution = configuration['input_resolution']
    
    def get_configuration(self):
        return {
                'input_resolution': self.input_resolution,
                'kernels_configuration': self.kernels_configuration,
                'depth_configuration': self.depth_configuration,
                'width_configuration': self.width_configuration,
                'sections_configuration':self.sections_configuration
                }
    
    def calculate_param_size(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                self.active_params += layer.active_params
        return self.active_params #*32 /8 /1000 /1000 # MB

    def calculate_feature_size(self):
        for name, layer in self.layers_dict.items():
            if isinstance(layer, ConvBlock):
                self.feature_size += layer.feature_size
        return self.feature_size #*32 /8 /1000 /1000 # MB

    def create_submodel_from_configuration(self, x):    
        
        layer_outputs = []
        upsampling_outputs = []
        res_i = 0

        for section in range(self.max_sections):
            if section in self.skip_sections:
                continue

            if section > self.max_sections // 2:
                # Up conv
                if self.distill_knowledge:
                    upsampling_outputs.append(x)

                # Upsampling
                x = self.layers_dict['conv3D_{}_{}'.format('up', self.max_sections - section)](x)
                
                if self.sampling_mode:
                    self.feature_size += np.prod(x.size())
                
                # Concat layer
                if self.distill_knowledge:
                    x = torch.cat((layer_outputs.pop(), x), 1)
                else:
                    x = torch.cat((layer_outputs[-1 - res_i], x), 1)
                    res_i += 1

            for i in range(self.depth_configuration[section]):
                out_channel = self.width_configuration[self.max_depth * section + i]
                self.layers_dict['conv3D_sec_{}_conv_{}'.format(section + 1, i + 1)].set_current_kernel(self.kernels_configuration[self.max_depth * section + i])
                self.layers_dict['conv3D_sec_{}_conv_{}'.format(section + 1, i + 1)].set_in_out_channel(x.size(1), out_channel)
                x = self.layers_dict['conv3D_sec_{}_conv_{}'.format(section + 1, i + 1)](x)
            
            if section < self.max_sections // 2:
                # Skip connection
                layer_outputs.append(x)

                # Maxpool
                x = self.layers_dict['conv3D_{}_{}'.format('down', section+1)](x)
                if self.sampling_mode:
                    self.feature_size += np.prod(x.size())
                
            
        # ---------------- Output ------------------
        out_channel = self.width_configuration[-1]
        self.layers_dict['conv3D_final_output'].set_current_kernel(self.last_kernel_configuration)
        self.layers_dict['conv3D_final_output'].set_in_out_channel(x.size(1), out_channel)
        x = self.layers_dict['conv3D_final_output'](x)
        if self.distill_knowledge:
            return x, layer_outputs + upsampling_outputs
        else:
            return x

    def forward(self, x):
        return self.create_submodel_from_configuration(x)
        
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
