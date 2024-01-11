import sys

import hdf5storage
import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm

from model import FullConnectedNet

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
    Y = Y.reshape(-1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_tensor = torch.tensor(X, dtype=torch.float32)
    label_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(data_tensor, label_tensor)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validate_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    val_num = len(test_set)

    # Create model
    net = FullConnectedNet()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    epochs = 100

    # load model weights
    save_path = "./FullConnectedNet.pth"
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels - 1
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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
                val_labels = val_labels - 1
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print("Finished Training. The best accuracy is: {}".format(best_acc))