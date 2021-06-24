#CJ -s '../../data' '/scratch/users/papyan/datasets'

import torch
import itertools

from exper.experiment import Experiment
from utils.accuracy import compute_accuracy
from utils.dataset_properties import get_dataset_properties

dataset_list    = [
                    'MNIST',
                    'FashionMNIST',
                    'CIFAR10',
                    'CIFAR100',
                    'STL10',
                    'SVHN',
                   ]

net_list        = [
                    'MLP',
                    'LeNet',
                   ]

lr_list           = [
                     0.5,
                     0.1,
                     0.05,
                     0.01,
                     ]
                     
X = len(dataset_list)
Y = len(net_list)
Z = len(lr_list)
for dataset_idx, net_idx, lr_idx in itertools.product(range(X), range(Y), range(Z)):
            im_size, num_classes, input_ch, size_dataset = \
                get_dataset_properties(dataset_list[dataset_idx])

            loader_opts  = {
                            'dataset'           : dataset_list[dataset_idx],
                            'loader_type'       : 'Natural',
                            'pytorch_dataset'   : True,
                            'dataset_path'      : '../../data',
                            'im_size'           : im_size,
                            'padded_im_size'    : im_size,
                            'num_classes'       : num_classes,
                            'input_ch'          : input_ch,
                            'threads'           : 4,
                            'epc_seed'          : 0,
                            }

            optim_kwargs = {
                            'weight_decay'      : 5e-4,
                            'momentum'          : 0.9,
                            }

            train_opts   = {
                            'crit'              : 'CrossEntropyLoss',
                            'net'               : net_list[net_idx],
                            'optim'             : 'SGD',
                            'optim_kwargs'      : optim_kwargs,
                            'epochs'            : 2,
                            # 'epochs'            : 20,
                            'lr'                : lr_list[lr_idx],
                            'milestones'        : [],
                            'gamma'             : 0.1,
                            'train_batch_size'  : 2**7,
                            'test_batch_size'   : 2**9,
                            'device'            : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            'seed'              : 0,
                            }

            results_opts  = {
                            'training_results_path': './results',
                            'train_dump_file'   : 'training_results.json',
                            }

            opts = dict(loader_opts, **train_opts)
            opts = dict(opts, **results_opts)

            def compute_accuracy_aux(variables, k):
                return compute_accuracy(variables['est'].data, variables['target'].data, topk=(k,))[0][0]

            stats_meter    = {
                                'top1' : lambda variables: float(compute_accuracy_aux(variables, 1).item()),
                                'loss' : lambda variables: float(variables['loss'].item()),
                            }

            stats_no_meter = {}

            exp = Experiment(opts)

            exp.run(stats_meter, stats_no_meter)
