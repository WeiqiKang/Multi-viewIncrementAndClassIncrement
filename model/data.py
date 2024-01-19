import os

import numpy as np
import hdf5storage
from torch.utils.data import Dataset, ConcatDataset, Subset
from sklearn.preprocessing import MinMaxScaler

class Multi_view_data(Dataset):
    """
    load multiview dataset
    """

    def __init__(self, root):
        """
        :param root:  data name and path
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        data_path = root

        dataset = hdf5storage.loadmat(data_path)
        view_number = len(dataset['X'][0])  # 视图个数

        self.X = dict()

        for v_num in range(view_number):
            self.X[v_num] = normalize(dataset['X'][0][v_num])
        y = dataset['Y']

        if np.min(y) == 1:
            y = y - 1  # 让下标从0开始
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        """
        获取第i个样本的所有信息。返回值data是一个字典，其key是视图索引。target是第i个样本的标签
        因为该函数的返回值是一个字典，dataloader会将具有相同键的值堆叠起来
        """
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x

def get_class_i(dataset, class_i, y_intervals):
    """
    获取第i个类的训练集、测试集
    :param dataset: 要分割的数据集
    :param class_i: 获取的类i
    :return: 第i个类的训练集、测试集
    """

    start_index = y_intervals[class_i][0]
    end_index = y_intervals[class_i][1]
    train_indices = list(range(len(dataset)))[start_index: int(start_index + (end_index - start_index + 1) * 0.8)]
    test_indices = list(range(len(dataset)))[int(start_index + (end_index - start_index + 1) * 0.8): end_index + 1]

    class_i_train_dataset = Subset(dataset, train_indices)
    class_i_test_dataset = Subset(dataset, test_indices)

    return class_i_train_dataset, class_i_test_dataset

def split_dataset_by_class(dataset):
    """
    按类分割数据集，每个阶段的类不相互重叠。训练集80%，测试集20%
    :param y_intervals: 每个类别y对应的区间
    :param dataset_name: 数据集名称
    :param dataset: 所要分割的数据集
    :return: 2个值，其一是一个列表，第i个值代表第i个类的训练集
    另一个是测试数据集，包含上述所有阶段的类
    """
    y_labels = [y for _, y in dataset] # 获取所有的y标签
    y_labels_size = len(set(y_labels))  # 求出有多少个类
    y_intervals = dict()

    # 遍历列表，记录每个值的起始和结束索引
    for i, label in enumerate(y_labels):
        if label not in y_intervals:
            y_intervals[int(label)] = [i, i]
        else:
            y_intervals[int(label)][1] = i

    print(y_intervals)

    test_set = None  # 分割后的测试数据集，要求包含所有类
    train_set_by_class = []

    for i in range(y_labels_size):
        tmp = get_class_i(dataset, i, y_intervals)
        train_set_by_class.append(tmp[0])
        if test_set is not None:
            test_set = ConcatDataset([test_set, tmp[1]])
        else:
            test_set = tmp[1]

    return train_set_by_class, test_set