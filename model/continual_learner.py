import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

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

class EWC(object):
    def __init__(self, model, dataset, epoch, batch_size, cuda_device):
        self.model = model
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.cuda_device = cuda_device
        self.epoch = epoch
        self.fisher = self.compute_fisher()


    def compute_fisher(self):
        fisher = {}
        for n, p in self.params.items():
            fisher[n] = torch.zeros_like(p.data)

        self.model.eval()
        loss_meter = AverageMeter()

        for batch_idx, (data, target) in enumerate(self.dataloader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda(device=self.cuda_device))
            target = Variable(target.long().cuda(device=self.cuda_device))
            self.model.zero_grad()
            evidence, evidence_a, loss = self.model(data, target, self.epoch)
            loss.backward()

            for n, p in self.params.items():
                fisher[n] += p.grad ** 2
            loss_meter.update(loss.item())

        # Average over the dataset
        fisher = {n: f / (len(self.dataloader) * self.batch_size) for n, f in fisher.items()}
        return fisher


    def ewc_loss(self, lambda_ewc):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return lambda_ewc * loss