from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from model.data import *
from model.model import TMC
import torch
import time

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.args = set_up_datasets(self.args)
        self.set_up_model()

    def set_up_model(self): # 设置当前训练类的model
        args = self.args
        self.model = TMC(args.num_classes, args.views, args.dims, args.lambda_epochs)
        self.model = nn.DataParallel(self.model, list(range(args.num_gpu)))
        self.model = self.model.cuda()

        if args.model_dir != None:
            print('Loading init parameters from %s' % args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print("random init params")
            if args.start_session > 0:
                print("WARNING: Random init weights for new sessions!")
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_dataloader(self, session, train_set_by_class, test_set):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args, train_set_by_class, test_set)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session, train_set_by_class, test_set)

        return trainset, trainloader, testloader

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler
    
    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]
        
        # 完整的数据集
        self.full_dataset = Multi_view_data(args.data_path)

        for session in range(args.start_session, args.sessions):
            # 分割数据集，按类别分割，保存在数组里
            train_set_by_class, test_set = split_dataset_by_class(dataset=self.full_dataset, session=session)
            train_set, trainloader, testloader = self.get_dataloader(session, train_set_by_class, test_set)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:
                print('new classes for this session:\n', list(range(60)))
                optimizer, scheduler = self.get_optimizer_base()
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # 在上面取出的base数据集上训练
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)