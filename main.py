import os
import torch
from datetime import datetime
from models.handler import train, test
import argparse
import numpy as np


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='ECG_data')
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--step_type', type=str, required=True)
    parser.add_argument('--step_cl', type=int, default=2500)
    parser.add_argument('--train_length', type=float, default=7)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--multi_layer', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--norm_method', type=str, required=True)
    parser.add_argument('--optimizer', type=str, default='RMSProp')
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.7)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--leakyrelu_rate', type=float, default=0.2)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--specific', type=str, default='none')
    parser.add_argument('--region', type=str, default='none')
    args = parser.parse_args()
    return args


def main():
    args = parsing_args()
    print(f'Training configs: {args}')

    data_file = f'dataset/{args.dataset}.npz'
        
    result_train_file = os.path.join('output', args.dataset + args.specific + '_' + args.region, 'train')
    result_test_file = os.path.join('output', args.dataset + args.specific + '_' + args.region, 'test')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)

    data = np.load(data_file)['data'][..., 0]

    args.train_length = 6
    args.valid_length = 2
    args.test_length = 2
    train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
    valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    test_ratio = 1 - train_ratio - valid_ratio
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    print(f'train: {len(train_data)}')
    print(f'valid: {len(valid_data)}')
    print(f'test: {len(test_data)}')
    print(f'total date: {len(data)/(12*24)}')
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, test_data, args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate and len(test_data) > 0:
        test(test_data, args, result_train_file, result_test_file)
    print('done')


if __name__ == '__main__':
    main()
