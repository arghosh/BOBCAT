import argparse
import numpy as np
import torch
import random


def initialize_seeds(seedNum):
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)


def create_parser():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--model', type=str,
                        default='biirt-random', help='type')
    parser.add_argument('--name', type=str, default='demo', help='type')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='type')
    parser.add_argument('--question_dim', type=int, default=4, help='type')
    parser.add_argument('--lr', type=float, default=1e-4, help='type')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='type')
    parser.add_argument('--inner_lr', type=float, default=1e-1, help='type')
    parser.add_argument('--inner_loop', type=int, default=5, help='type')
    parser.add_argument('--policy_lr', type=float, default=2e-3, help='type')
    parser.add_argument('--dropout', type=float, default=0.5, help='type')
    parser.add_argument('--dataset', type=str,
                        default='assist2009', help='eedi-1 or eedi-3')
    parser.add_argument('--fold', type=int, default=1, help='type')
    parser.add_argument('--n_query', type=int, default=10, help='type')
    parser.add_argument('--seed', type=int, default=221, help='type')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    params = parser.parse_args()
    #
    if params.dataset == 'eedi-3':
        params.n_question = 948
        params.train_batch_size = 512
        params.test_batch_size = 1000
        params.n_epoch = 10000
        params.wait = 1000
        params.repeat = 5
    if params.dataset == 'eedi-1':
        params.n_question = 27613
        params.n_epoch = 750
        params.train_batch_size = 128
        params.test_batch_size = 512
        params.wait = 50
        params.repeat = 2
    if params.dataset == 'assist2009':
        params.n_question = 26688
        params.train_batch_size = 128
        params.test_batch_size = 512
        params.n_epoch = 5000
        params.wait = 250
        params.repeat = 10
    if params.dataset == 'ednet':
        params.n_question = 13169
        params.n_epoch = 500
        params.train_batch_size = 200
        params.test_batch_size = 512
        params.wait = 25
        params.repeat = 1
    if params.dataset == 'junyi':
        params.n_question = 25785
        params.n_epoch = 750
        params.train_batch_size = 128
        params.test_batch_size = 512
        params.wait = 50
        params.repeat = 2

    return params
