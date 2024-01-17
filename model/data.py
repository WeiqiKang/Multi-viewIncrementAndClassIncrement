import os

import numpy as np
import hdf5storage
from torch.utils.data import Dataset
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


