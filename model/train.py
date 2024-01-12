import sys

import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm
from read_data import read_data
from model import FullConnectedNet, CosineClassifier


def train_one_task(model, dataset):
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    validate_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    val_num = len(test_set)

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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 读取数据集
    X, Y = read_data(dataset="100Leaves")
    Y = Y.reshape(-1)

    # 将数据集转换为tensor
    data_tensor = torch.tensor(X, dtype=torch.float32)
    label_tensor = torch.tensor(Y, dtype=torch.long)

    # 打包数据集
    dataset = TensorDataset(data_tensor, label_tensor)

    # 定义分类器，采用全连接分类器
    net = FullConnectedNet(in_features=64, n_classes=100)
    net.to(device)

    # 定义阶段个数
    num_tasks = 5

    # 分割数据集，将数据集随机分成num_tasks份
    total_size = len(dataset)
    subset_size = total_size // num_tasks
    sizes = [subset_size] * (num_tasks - 1)
    sizes.append(total_size - sum(sizes))
    subsets = random_split(dataset, sizes)  # 此时subsets是一个列表，包含了10个随机分割的子数据集，每个子数据集都是TensorDataset类型

    # 对每个阶段进行训练
    for task in range(num_tasks):
        print("Task {}/{}".format(task+1, num_tasks))
        train_one_task(net, subsets[task])
        print("Finished Task {}/{}".format(task+1, num_tasks))
