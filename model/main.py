import sys

import hdf5storage
import os
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model import AlexNet
from torch.utils.data import TensorDataset, DataLoader, random_split


def read_data(dataset):
    # 数据集名称和路径
    data_name = dataset + ".mat"
    data_base_path = os.path.join(os.getcwd(), "..", 'data')
    data_path = os.path.join(data_base_path, data_name)

    # 加载数据集
    data = hdf5storage.loadmat(data_path)
    # data['X'][0][0]是第一个视图，data['X'][0][1]是第二个视图,...,以此类推
    view_1 = data['X'][0][0]
    view_2 = data['X'][0][1]
    view_3 = data['X'][0][2]
    Y = data['Y']

    return view_1, Y  # 暂时先用一个视图试水


if __name__ == "__main__":
    X, Y = read_data(dataset="100Leaves")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_tensor = torch.tensor(X, dtype=torch.float32)
    label_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(data_tensor, label_tensor)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validate_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Create model
    net = AlexNet(64, 100, 100)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    loss_function = nn.CrossEntropyLoss()
    epochs = 10
    # load model weights
    save_path = "./AlexNet.pth"
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # images = data_transform(images)
            print(images.shape)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            print(outputs)
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 设置进度条的描述信息
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                print(outputs)