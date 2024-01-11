import os

import hdf5storage


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