import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from data import Multi_view_data
from model import TMC

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


def train_one_task(model, train_dataset, test_dataset, ewc, num_task):
    # 定义Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
                loss += ewc.ewc_loss(lambda_ewc=0.5)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda(device=cuda_device))
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long().cuda(device=cuda_device))
                evidences, evidence_a, loss = model(data, target, epoch)
                _, predicted = torch.max(evidence_a.data, 1)
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())
        print('====> acc: {:.4f}'.format(correct_num / data_num))
        return loss_meter.avg, correct_num / data_num

    # 定义进度条

    for epoch in tqdm(range(1, args.epochs + 1)):
        train(epoch)

    test(epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
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
    cuda_device = 6
    model.cuda(device=cuda_device)

    # 定义阶段个数，为1时表示单阶段
    num_tasks = 5

    # 分割数据集，将数据集随机分成num_tasks份
    total_size = len(full_dataset)
    subset_size = total_size // num_tasks
    sizes = [subset_size] * (num_tasks - 1)
    sizes.append(total_size - sum(sizes))
    subsets = random_split(full_dataset, sizes)  # 此时subsets是一个列表，包含了num_tasks个随机分割的子数据集

    # 持续学习EWC
    ewc = None

    # 对每个阶段进行训练
    for task in range(num_tasks):
        # 定义训练集和测试集的大小
        train_size = int(0.8 * len(subsets[task]))
        test_size = len(subsets[task]) - train_size

        # 分割数据集
        train_dataset, test_dataset = random_split(subsets[task], [train_size, test_size])

        # 训练
        print("Task {}/{}".format(task + 1, num_tasks))
        train_one_task(model, train_dataset, test_dataset, ewc, task)
        print("Finished Task {}/{}".format(task + 1, num_tasks))

        # 更新持续学习EWC
        ewc = EWC(model, train_dataset, args.epochs, args.batch_size, cuda_device)


    print(test_acc)
