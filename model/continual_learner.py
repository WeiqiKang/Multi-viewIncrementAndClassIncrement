import torch
import torch.nn as nn


class EWC(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher()

    def compute_fisher(self):
        fisher = {}
        for n, p in self.params.items():
            fisher[n] = torch.zeros_like(p.data)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        for input, target in self.dataset:
            self.model.zero_grad()
            output = self.model(input)
            loss = criterion(output, target)
            loss.backward()

            for n, p in self.params.items():
                fisher[n] += p.grad ** 2

        # Average over the dataset
        fisher = {n: f / len(self.dataset) for n, f in fisher.items()}
        return fisher

    def ewc_loss(self, lambda_ewc):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return lambda_ewc * loss