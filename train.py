import os
import argparse
from utils import *
import importlib

import torch
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, ConcatDataset
from tqdm import tqdm
from model.data import Multi_view_data
from model.model import TMC
from model.data import split_dataset_by_class
from model.continual_learner import EWC

MODEL_DIR = None
PROJECT = 'base'
DATA_DIR = 'data'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='100Leaves',
                        choices=['100Leaves', 'Aloi-100', 'Animal', 'bbcsport', 'Caltech-5V'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-batch_size_base', type=int, default=64, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=64)

    parser.add_argument('-lambda_epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-start_session', type=int, default=0)
    
    # about hardware
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    
    return parser

if __name__ == "__main__":

    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('model.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()

# python train.py -project base -dataset 100Leaves  -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 60 70 -gpu 0,1,2,3 -temperature 16
