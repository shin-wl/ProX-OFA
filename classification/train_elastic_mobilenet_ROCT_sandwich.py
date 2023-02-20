import os
from elastic_mobilenet import ElasticMobileNet
from validation_utils import calibrate_batchnorm, accuracy_metric, accuracy_metric_numpy, feature_layer_loss, output_layer_loss, validate
from dataset import ROCTDataset
from logger import Logger, stage_to_string
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import math
import json
import timeit
import datetime
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter

def set_largest_configuration(net, train_generator):
    # largest net
    train_generator.dataset.set_input_dim(randomize=True, 
                                          permute_from=[max(choices['resolution_choices'])])
    net.input_resolution = train_generator.dataset.input_dim
    net.set_kernels_configuration(randomize=True, 
                                  permute_from=[max(choices['kernels_choices'])])
    net.depth_configuration = [random.choice([max(choices['depth_choices'])]) 
                               for _ in range(net.sections_configuration)]
    net.set_width_expansion_configuration(randomize=True, 
                                          permute_from=[max(choices['expansion_choices'])]) 
    net.set_width_configuration(randomize=False, stage=1)
    return net, train_generator

def set_smallest_subnet_configuration(net):
    # largest net
    # train_generator.dataset.set_input_dim(randomize=True, 
    #                                       permute_from=[min(choices['resolution_choices'])])
    # net.input_resolution = train_generator.dataset.input_dim
    net.input_resolution = min(choices['resolution_choices'])
    net.set_kernels_configuration(randomize=True, 
                                  permute_from=[min(choices['kernels_choices'])])
    net.depth_configuration = [random.choice([min(choices['depth_choices'])]) 
                               for _ in range(net.sections_configuration)]
    net.set_width_expansion_configuration(randomize=True, 
                                          permute_from=[min(choices['expansion_choices'])]) 
    net.set_width_configuration(randomize=False, stage=1)
    return net

def set_random_subnet_configuration(net):
    # largest net
    # train_generator.dataset.set_input_dim(randomize=True, 
    #                                       permute_from=[choices['resolution_choices']])
    net.input_resolution = random.choice(choices['resolution_choices'])
    net.set_kernels_configuration(randomize=True, 
                                  permute_from=choices['kernels_choices'])
    net.depth_configuration = [random.choice(choices['depth_choices']) 
                               for _ in range(net.sections_configuration)]
    net.set_width_expansion_configuration(randomize=True, 
                                          permute_from=choices['expansion_choices']) 
    net.set_width_configuration(randomize=False, stage=1)
    return net

if __name__ == '__main__':

    with open('config-mobilenet-ROCT.json') as f:
        config = json.load(f)

    init_config = config['initialization']
    ofa_settings = config['ofa_settings']


    labels = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
    EPOCHS = 1
    BATCH_SIZE = 48
    SOFT_BATCH = 8

    input_dim = 224

    train_set = ROCTDataset(labels=labels, dataset_type='train', input_dim=input_dim, data_dir='data/Retinal OCT')
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    val_set = ROCTDataset(labels=labels, dataset_type='val', input_dim=input_dim, data_dir='data/Retinal OCT')
    val_generator = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    test_set = ROCTDataset(labels=labels, dataset_type='test', input_dim=input_dim, data_dir='data/Retinal OCT')
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=8, num_workers=4, pin_memory=True)

    n_train = math.ceil(len(train_set)// BATCH_SIZE)
    n_test = len(test_set) // 8

    iterations_per_stage = math.ceil(len(train_generator)/ SOFT_BATCH) * ofa_settings['epoch_per_stage']
    iterations_per_stage



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Params Elastic Kernel
    expanding_kernels=init_config['expanding_kernels']

    # Params Elastic Width
    keep_channel_order=init_config['keep_channel_order']


    input_dim=init_config['input_dim'] 
    output_n=init_config['output_n']
    kernels=init_config['kernels']
    max_depth=init_config['max_depth']
    sections=init_config['sections']

    net = ElasticMobileNet(input_dim=input_dim, 
                        output_n=output_n, 
                        max_depth=max_depth, 
                        expanding_kernels=expanding_kernels,
                        keep_channel_order=keep_channel_order,
                        kernels=kernels,
                        sections=sections,
                        bn_track_running_stats=True).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=init_config['lr']) 
    loss_output_func = output_layer_loss


    mode = ofa_settings['mode']
    stage_name = ofa_settings['stage_name']

    model_name = '{}-{}'.format(mode, stage_name) if stage_name != '' else mode

    print(model_name)
    print('lr: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

    if mode == 'expand':
        choices = {
            'depth_choices': sorted(ofa_settings['choices']['depth_choices']),
            'expansion_choices': sorted(ofa_settings['choices']['expansion_choices']),
            'kernels_choices': sorted(ofa_settings['choices']['kernels_choices']),
        }
    if mode == 'ps':
        choices = {
            'kernels_choices': sorted(ofa_settings['choices']['kernels_choices'], reverse=True),
            'depth_choices': sorted(ofa_settings['choices']['depth_choices'], reverse=True),
            'expansion_choices': sorted(ofa_settings['choices']['expansion_choices'], reverse=True),
        }

    if not ofa_settings['mixed_resolution']:
        choices['resolution_choices'] = ofa_settings['choices']['resolution_choices']

    initial_configurations = {k: [min(v)] for k, v in choices.items()} if mode == 'expand' else {k: [max(v)] for k, v in choices.items()}

    if ofa_settings['mixed_resolution']:
        initial_configurations['resolution_choices'] = ofa_settings['choices']['resolution_choices']

    initial_configurations['repeated'] = 0
    stages_configurations = [copy.deepcopy(initial_configurations)]


    for k,v in choices.items():
        print(k,v)
        for item in v[1:]:
            configurations = copy.deepcopy(stages_configurations[-1])
            configurations[k].append(item)
            stages_configurations += [configurations]


    # Extended Training
    for i in range(1, ofa_settings['extend_training'] + 1,):
        configurations = copy.deepcopy(stages_configurations[-1])
        configurations['repeated'] = i
        stages_configurations += [configurations]

    i = 0
    for c in stages_configurations:
        print(i, c)
        i += 1


    if init_config['pretrained_logger_dir']:
        with open(init_config['pretrained_logger_dir'], 'rb') as f:
            logger = pickle.load(f)
    else:
        logger = Logger()


    writer = SummaryWriter()
    run_name = config['ofa_settings']['save_weights_dir'].split('/')[-1]

    n_iter = 0
    stage_to_string(2, stages_configurations)

    print(len(stages_configurations))

    epoch_per_stage = ofa_settings['epoch_per_stage']
    print(epoch_per_stage)


    # Sandwich rules
    iter_time = 0
    time_elapsed = 0
    patience = 0
    last_best_iter = [None, None]
    average_sample_size = 5
    average_sample_count = 0
    val_loss = np.float32(0)
    val_score = 0
    total_epochs = epoch_per_stage * len(stages_configurations)

    logger.start_new_stage()

    for e in range(total_epochs):

        output_distill_knowledge = False if e < 5 else True

        # iter_loss = 0
        # iter_ce_loss = 0
        all_loss = 0
        optimizer.zero_grad()
        start_time = timeit.default_timer()
        count = 1
        n_per_batch = SOFT_BATCH
        with tqdm(train_generator, unit="iter") as tepoch:
            # for X, y in tepoch:
                # tepoch.set_description("Epoch {}".format(e))
            for X, y in train_generator:
                X, y = X.to(device), y.to(device)
                
                net, train_generator = set_largest_configuration(net, train_generator)
                # print('largest', net.get_configuration())
                largest_net_output = net(X)


                # N=2 random subnet
                net = set_random_subnet_configuration(net)
                subnet_resolution = net.input_resolution
                # print('random_1', net.get_configuration())
                X = F.interpolate(X, size=(subnet_resolution, subnet_resolution))
                subnet_output_1 = net(X)
                val_net = copy.deepcopy(net)

                net = set_random_subnet_configuration(net)
                subnet_resolution = net.input_resolution
                # print('random_2', net.get_configuration())
                X = F.interpolate(X, size=(subnet_resolution, subnet_resolution))
                subnet_output_2 = net(X)
                
                net = set_smallest_subnet_configuration(net)
                subnet_resolution = net.input_resolution
                # print('smallest', net.get_configuration())
                X = F.interpolate(X, size=(subnet_resolution, subnet_resolution))
                smallest_net_output = net(X)

                largest_net_loss = loss_func(largest_net_output, y) / n_per_batch
                
                subnet_loss_1 = (loss_func(subnet_output_1, y) / n_per_batch + loss_output_func(subnet_output_1, largest_net_output) / n_per_batch) / 2
                
                subnet_loss_2 = (loss_func(subnet_output_2, y) / n_per_batch + loss_output_func(subnet_output_2, largest_net_output) / n_per_batch) / 2
                
                smallest_net_loss = (loss_func(smallest_net_output, y) / n_per_batch + loss_output_func(smallest_net_output, largest_net_output) / n_per_batch) / 2

                loss = (largest_net_loss + subnet_loss_1 + subnet_loss_2 + smallest_net_loss) / 4

                print(loss)

                loss.backward()

                # iteration summary
                if count % SOFT_BATCH == 0 or count >= n_train:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    n_per_batch = (n_train % SOFT_BATCH) if ((n_train - count) < SOFT_BATCH) else SOFT_BATCH

                    iter_time = timeit.default_timer() - start_time
                    time_elapsed += iter_time

                    print('Validating...')
                    # val_net = copy.deepcopy(net)
                    calibrate_batchnorm(val_net, val_generator)
                    val_score, val_loss = validate(val_net, val_generator, accuracy_metric_numpy, loss_func, device)
                    del val_net

                    start_time = timeit.default_timer()

                    # all_loss += iter_loss

                    logger.record(time_elapsed=datetime.timedelta(seconds=time_elapsed), val_score=val_loss, train_loss=loss)
                    # tepoch.set_postfix(iter_loss=iter_loss, avg_loss=all_loss/(count/SOFT_BATCH), val_loss=val_loss, accuracy=val_score)
                    print(val_score)
                    writer.add_scalar('{}/train'.format(run_name), loss, n_iter)
                    writer.add_scalar('{}/val'.format(run_name), val_loss, n_iter)
                    writer.add_scalar('{}/accuracy'.format(run_name), val_score, n_iter)
                    n_iter += 1
                    # break

                # iter_ce_loss = 0
                # iter_loss = 0

                count += 1
        with open('{}/logger-{}-{}.pkl'.format(ofa_settings['save_log_dir'], model_name, e), 'wb') as f:
            pickle.dump(logger, f)
        break

# for stage in range(0, len(stages_configurations)):


#     if stage >= 1:
#         train_generator.dataset.set_input_dim(randomize=True, 
#                                               permute_from=[max(stages_configurations[stage - 1]['resolution_choices'])])
#         net.input_resolution = train_generator.dataset.input_dim
#         net.set_kernels_configuration(randomize=True, 
#                                       permute_from=[max(stages_configurations[stage - 1]['kernels_choices'])])
#         net.depth_configuration = [random.choice([max(stages_configurations[stage - 1]['depth_choices'])]) 
#                                    for _ in range(net.sections_configuration)]
#         net.set_width_expansion_configuration(randomize=True, 
#                                               permute_from=[max(stages_configurations[stage - 1]['expansion_choices'])]) 
#         net.set_width_configuration(randomize=False, stage=1)
#     best_configuration = copy.deepcopy(net.get_configuration())

#     output_distill_knowledge = False if stage < 1 else True

#     if output_distill_knowledge:
#         copy_net = copy.deepcopy(net)
#         copy_net.load_configuration(best_configuration)
    
#     print('Stage: {}'.format(stage))
#     print(stages_configurations[stage])
#     logger.start_new_stage()
    
#     if ofa_settings['sort_channel']:
#         if len(stages_configurations[stage]['expansion_choices']) > 1 and stages_configurations[stage]['repeated'] == 0:
#             print('sorted channels')
#             net.sort_all_channels(conv_type='depthwise_conv')


#     iter_time = 0
#     time_elapsed = 0
#     patience = 0
#     last_best_iter = [None, None]
#     average_sample_size = 5
#     average_sample_count = 0
#     val_loss = np.float32(0)
#     val_score = 0

#     for e in range(epoch_per_stage):

#         iter_loss = 0
#         iter_ce_loss = 0
#         all_loss = 0
#         optimizer.zero_grad()
#         start_time = timeit.default_timer()
#         count = 1
#         n_per_batch = SOFT_BATCH
#         with tqdm(train_generator, unit="iter") as tepoch:
#             for X, y in tepoch:
#                 tepoch.set_description("Stage {} Epoch {}".format(stage, e))

#                 X, y = X.to(device), y.to(device)

#                 # EDIT HERE
#                 # Elastic Depth
#                 if (mode == 'expand' and stage_name == 'elastic-depth') or (mode == 'expand' and ofa_settings['train_full_dimension']):
#                     if count % SOFT_BATCH == 0 or count >= n_train:
#                         net.set_depth_configuration(permute_from=stages_configurations[stage]['depth_choices'], freeze_output_layer=False)
#                 else:
#                     net.depth_configuration = [random.choice(stages_configurations[stage]['depth_choices']) 
#                                                for _ in range(net.sections_configuration)]



#                 # Elastic Kernel
#                 if (mode == 'expand' and stage_name == 'elastic-kernels') or (mode == 'expand' and ofa_settings['train_full_dimension']):
#                     if count % SOFT_BATCH == 0 or count >= n_train:
#                         net.set_kernels_configuration(randomize=True, permute_from=stages_configurations[stage]['kernels_choices'])
#                 else:
#                     net.set_kernels_configuration(randomize=True, 
#                                               permute_from=stages_configurations[stage]['kernels_choices'])


#                 # Elastic Width
#                 net.set_width_configuration(randomize=False, stage=1)

#                 net.set_width_expansion_configuration(randomize=True, 
#                                                       permute_from=stages_configurations[stage]['expansion_choices'])
#                 # Elastic resolution
#                 net.input_resolution = train_generator.dataset.input_dim
#                 train_generator.dataset.set_input_dim(randomize=True, 
#                                                       permute_from=stages_configurations[stage]['resolution_choices'])
                


#                 output = net(X)

#                 loss = loss_func(output, y) / n_per_batch

#                 if output_distill_knowledge:
#                     # For logging purpose
#                     iter_ce_loss += loss_func(output, y) / n_per_batch

#                     with torch.no_grad():
#                         kd_resolution = copy_net.input_resolution
#                         X = F.interpolate(X, size=(kd_resolution, kd_resolution))
#                         teacher_output = copy_net(X)
                    
#                     loss += loss_output_func(output, teacher_output) / n_per_batch
#                     loss /= 2

#                 loss.backward()

#                 if net.expanding_kernels:
#                     net.freeze_all_middle_kernel_grads()


#                 iter_loss += loss

#                 # iteration summary
#                 if count % SOFT_BATCH == 0 or count >= n_train:
#                     n_per_batch = (n_train % SOFT_BATCH) if ((n_train - count) < SOFT_BATCH) else SOFT_BATCH

#                     optimizer.step()
#                     optimizer.zero_grad()

#                     iter_time = timeit.default_timer() - start_time
#                     time_elapsed += iter_time

#                     val_net = copy.deepcopy(net)
#                     calibrate_batchnorm(val_net, val_generator)
#                     val_score, val_loss = validate(val_net, val_generator, accuracy_metric_numpy, loss_func, device)
#                     del val_net

#                     start_time = timeit.default_timer()

#                     all_loss += iter_loss


#                     if distill_knowledge or output_distill_knowledge:
#                         logger.record(time_elapsed=datetime.timedelta(seconds=time_elapsed), val_score=val_loss, train_loss=iter_ce_loss, distill_knowledge=distill_knowledge, other_loss=iter_loss)
#                         tepoch.set_postfix(iter_loss=iter_ce_loss, avg_loss=all_loss/(count/SOFT_BATCH), val_loss=val_loss ,accuracy=val_score)
#                         writer.add_scalar('{}/train'.format(run_name), iter_ce_loss, n_iter)
#                         writer.add_scalar('{}/val'.format(run_name), val_loss, n_iter)
#                         writer.add_scalar('{}/accuracy'.format(run_name), val_score, n_iter)
#                         n_iter += 1
#                     else:
#                         logger.record(time_elapsed=datetime.timedelta(seconds=time_elapsed), val_score=val_loss, train_loss=iter_loss)
#                         tepoch.set_postfix(iter_loss=iter_loss, avg_loss=all_loss/(count/SOFT_BATCH), val_loss=val_loss, accuracy=val_score)
#                         writer.add_scalar('{}/train'.format(run_name), iter_loss, n_iter)
#                         writer.add_scalar('{}/val'.format(run_name), val_loss, n_iter)
#                         writer.add_scalar('{}/accuracy'.format(run_name), val_score, n_iter)
#                         n_iter += 1


#                     iter_ce_loss = 0
#                     iter_loss = 0

#                 count += 1

#     string_name = stage_to_string(stage, stages_configurations)
#     with open('{}/logger-{}-{}.pkl'.format(ofa_settings['save_log_dir'], model_name, string_name), 'wb') as f:
#         pickle.dump(logger, f)
#     torch.save(net.state_dict(), '{}/weights-{}-{}.pt'.format(ofa_settings['save_weights_dir'], model_name, string_name))
#     torch.save(optimizer.state_dict(), '{}/optimizer-{}-{}.pt'.format(ofa_settings['save_weights_dir'], model_name, string_name))
    
#     if output_distill_knowledge:
#         del copy_net