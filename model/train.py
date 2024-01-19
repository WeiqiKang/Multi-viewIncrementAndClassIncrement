import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, ConcatDataset
from tqdm import tqdm
from data import Multi_view_data
from model import TMC
from data import split_dataset_by_class
from continual_learner import EWC

test_acc = []


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def train_one_task(model, train_dataset, ewc, num_task):
    """
    Train one task
    :param model: 核心算法模型
    :param train_dataset: 训练集
    :param ewc: 持续学习核心
    :param num_task: 当前所在阶段
    """
    # 定义Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    N_mini_batches = len(train_loader)
    print("Number of training batches: %d" % N_mini_batches)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda(device=cuda_device))
            target = Variable(target.long().cuda(device=cuda_device))
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(data, target, epoch)
            if num_task > 0:
                loss += ewc.ewc_loss(lambda_ewc=2)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


    for epoch in tqdm(range(1, args.epochs + 1)):
        train(epoch)




def test(test_dataset):
    """
    Test function for dataset of all classes
    :param test_dataset: 测试数据集，其包含所有类
    """
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda(device=cuda_device))
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda(device=cuda_device))
            evidences, evidence_a, loss = model(data, target, args.epochs)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
    print('====> acc: {:.4f}'.format(correct_num / data_num))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    # 数据集相关
    args.data_name = "100Leaves"
    args.data_path = os.path.join("..", "data", args.data_name + ".mat")
    args.dims = [[64], [64], [64]]  # 每个视图的维度
    args.views = len(args.dims)  # 视图个数

    # 完整的数据集
    full_dataset = Multi_view_data(args.data_path)

    # 定义模型
    model = TMC(100, args.views, args.dims, args.lambda_epochs)
    cuda_device = 4
    model.cuda(device=cuda_device)

    # 定义阶段个数，为1时表示单阶段
    num_tasks = 2

    # 分割数据集，测试集按类别分割，保存在一个数组里。训练集包含所有类
    train_set_by_class, test_set = split_dataset_by_class(dataset=full_dataset)

    # 持续学习EWC
    ewc = None
    #
    # 将数据集按阶段分割，每个阶段用哪几个类
    train_set_by_stage = [[i for i in range(50)], [i for i in range(50, 100)]]
    # for i in range(10):
    #     train_set_by_stage.append([j for j in range(i * 10, i * 10 + 10)])
    print(train_set_by_stage)
    """
    以Caltech101-7为例，其包含7个类，那么就将其分割为：
        阶段0：类0 类1
        阶段1：类2 类3
        阶段2：类4 类5 类6
    """

    # 对每个阶段进行训练
    for task in range(num_tasks):
        train_dataset = train_set_by_class[train_set_by_stage[task][0]]
        for class_num in train_set_by_stage[task]:
            train_dataset = ConcatDataset([train_dataset, train_set_by_class[class_num]])

        # 训练
        print("Task {}/{}".format(task + 1, num_tasks))
        print(f"Load classes: {train_set_by_stage[task]}")
        train_one_task(model, train_dataset, ewc, task)
        print("Finished Task {}/{}".format(task + 1, num_tasks))

        # 更新持续学习EWC
        ewc = EWC(model, train_dataset, args.epochs, args.batch_size, cuda_device)


    # 最后，在包含所有类的测试集上进行测试，输出分类准确率
    test(test_set)

