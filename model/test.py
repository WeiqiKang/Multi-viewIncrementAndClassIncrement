import torch

X = torch.arange(12).reshape((3, 4))

print(X)
print(torch.max(X, dim=1)[1])