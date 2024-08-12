import numpy as np

from dataset import *
from model import *


def mse(data1, data2):
    """
    计算两个二维数据集之间的均方误差（MSE）

    参数：
    - data1: 第一个二维数据集，numpy数组
    - data2: 第二个二维数据集，numpy数组

    返回值：
    - mse_value: 均方误差值
    """
    # 确保两个数据集形状相同
    assert data1.shape == data2.shape, "数据集形状不一致"

    # 计算差值
    diff = data1 - data2

    # 计算平方差值
    squared_diff = np.square(diff)

    # 计算均方误差
    mse_value = np.mean(squared_diff)

    return mse_value

ntuiot_dataset = NTUIoTRSSI_Dataset('rssi_position_data.txt')
ntuiot_dataset.raw_set = ntuiot_dataset.decrease_dataset(5)
ntuiot_dataset.filter_outliers(quantile_threshold=0.1)
data_dict = ntuiot_dataset.split_train_test(train_ratio=0.4, val_ratio=0.2, random_state=10)
# ntuiot_dataset.get_compact('mean')
#k3： 0.1  0.2   0.3
#50:44991 35816
#40:45085 34120
#10:43753 33815 29367
# 0:45274 35234

#k6: 0.1
#10: 43789

#k10: 0.1
#10: 45572

wknn_model = WKNN(ntuiot_dataset.train_set[:, :ntuiot_dataset.dim_coord], ntuiot_dataset.train_set[:, -ntuiot_dataset.dim_rssi:])
# wknn_model = WKNN(ntuiot_dataset.compact_set[:, :ntuiot_dataset.dim_coord], ntuiot_dataset.compact_set[:, -ntuiot_dataset.dim_rssi:])
error = []
for i in range(len(ntuiot_dataset.val_set)):
    rssi, position = ntuiot_dataset.__getitem__(i)
    result = wknn_model(rssi.numpy(), K=4)
    error.append(mse(position.numpy(), result))
error_mean = np.mean(error)
print(error_mean)
