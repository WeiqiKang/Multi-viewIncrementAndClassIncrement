import math

import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn import functional as F


class FullConnectedNet(nn.Module):
    def __init__(self, in_features, n_classes):
        super(FullConnectedNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CosineClassifier(nn.Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out



class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):  # num_classes表示类别个数
        super(AlexNet, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features_extractor(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 从第一维开始展平，类似于view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历当前网络的所有层
            if isinstance(m, nn.Conv2d):  # 判断m是不是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 判断m是不是全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

