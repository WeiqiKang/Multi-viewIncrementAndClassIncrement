import hdf5storage
import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    std = StandardScaler()
    X_std = std.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.3)

    svm_classification = SVC(kernel='linear')
    svm_classification.fit(X_train, Y_train)

    print(f"分类准确率：{svm_classification.score(X_test, Y_test)}")
