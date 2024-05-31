import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class NTUIoTRSSI_Dataset(Dataset):
    def __init__(self, path_dataset):
        super(NTUIoTRSSI_Dataset, self).__init__()
        self.dim_coord = 2
        self.dim_rssi = 6
        self.path = path_dataset
        self.raw_set = self.read_txt()
        self.compact_set = None

    def read_txt(self):
        eval_law = {'np.nan': None}
        data_record_raw = []
        with open(self.path, encoding='utf-8') as file_obj:
            lines_read = file_obj.readlines()
            for line_read in lines_read:
                data_raw_temp = eval(line_read, eval_law)
                data_record_raw.append(data_raw_temp)
        data_record = np.array(data_record_raw, dtype=float)
        return data_record

    def get_compact(self, method):
        if self.compact_set is None:
            rssi_mean = []
            coord_set = self.raw_set[:, :self.dim_coord]
            rssi_set = self.raw_set[:, -self.dim_rssi:]
            unique_coords = np.unique(coord_set, return_index=False, return_counts=False, axis=0)
            for coord_tmp in unique_coords:
                idx_tmp = np.where((coord_set == coord_tmp).all(-1))[0]
                if method == 'mean':
                    rssi_mean_tmp = rssi_set[idx_tmp, :].mean(axis=0)

                elif method == 'median':
                    rssi_mean_tmp = np.median(rssi_set[idx_tmp,:], axis=0)

                rssi_mean.append(rssi_mean_tmp)
            rssi_mean = np.array(rssi_mean)
            self.compact_set = np.hstack([unique_coords, rssi_mean])

    def decrease_dataset(self, loc_num):
        unique_coords = np.unique(self.raw_set[:, :self.dim_coord], axis=0)
        decreased_set = []
        interval = len(unique_coords) // loc_num
        sample_coords = unique_coords[::interval][:loc_num]
        for coord_tmp in sample_coords:
            idx_tmp = np.where((self.raw_set[:, :self.dim_coord] == coord_tmp).all(-1))[0]
            data_tmp = self.raw_set[idx_tmp]
            decreased_set.extend(data_tmp)
        return np.array(decreased_set)


    def split_train_test(self, train_ratio=0.6, val_ratio=0.4, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        self.train_set = []
        self.val_set = []
        self.test_set = []
        coord_set = self.filtered_set[:, :self.dim_coord]
        unique_coords = np.unique(coord_set, return_index=False, return_counts=False, axis=0)
        for coord_tmp in unique_coords:
            idx_tmp = np.where((coord_set == coord_tmp).all(-1))[0]
            np.random.shuffle(idx_tmp)
            train_size = int(len(idx_tmp) * train_ratio)
            train_idx = idx_tmp[:train_size]
            val_idx = idx_tmp[train_size: train_size + int(len(idx_tmp) * val_ratio)]
            test_idx = idx_tmp[train_size + int(len(idx_tmp) * val_ratio):]
            self.train_set.extend(self.filtered_set[train_idx])
            self.val_set.extend(self.filtered_set[val_idx])
            self.test_set.extend(self.filtered_set[test_idx])
        self.test_set = np.array(self.test_set)
        self.val_set = np.array(self.val_set)
        self.train_set = np.array(self.train_set)
        return {'database_position': self.train_set[:, :self.dim_coord], 'database_rssi': self.train_set[:, -self.dim_rssi:],
                'test_position': self.test_set[:, :self.dim_coord],  'test_rssi': self.test_set[:, -self.dim_rssi:],
                'val_position': self.val_set[:, :self.dim_coord],  'val_rssi': self.val_set[:, -self.dim_rssi:]}

    def filter_outliers(self, quantile_threshold=0.8):
        """
        对数据集中的异常值进行过滤

        参数：
        - quantile_threshold: 分位数阈值，默认为0.95
        """
        self.filtered_set = []
        unique_coords = np.unique(self.raw_set[:, :self.dim_coord], axis=0)
        for coord_tmp in unique_coords:
            idx_tmp = np.where((self.raw_set[:, :self.dim_coord] == coord_tmp).all(-1))[0]
            data_tmp = self.raw_set[idx_tmp]
            # 计算统计指标，这里选择中位数作为统计指标
            statistic = np.median(data_tmp[:, self.dim_coord:], axis=0)
            # 计算样本与统计指标的差值的绝对值
            deviations = np.abs(data_tmp[:, self.dim_coord:] - statistic)
            # 计算每个样本的总偏差，这里选择每个样本的总偏差的第 quantile_threshold 分位数作为阈值
            total_deviations = np.sum(deviations, axis=1)
            threshold = np.quantile(total_deviations, quantile_threshold)
            # 过滤掉偏差较大的样本
            filtered_data_tmp = data_tmp[total_deviations <= threshold]
            self.filtered_set.extend(filtered_data_tmp)
        self.filtered_set = np.array(self.filtered_set)




    def __len__(self):
        return len(self.raw_set)

    # def __getitem__(self, item):
    #     rssi_sample = self.raw_set[item]
    #     position = rssi_sample[:self.dim_coord].astype(float)
    #     rssi_array = rssi_sample[-self.dim_rssi:].astype(float)
    #
    #     position = torch.tensor(position, dtype=torch.float32)
    #     rssi_tensor = torch.tensor(rssi_array, dtype=torch.float32)
    #
    #     return rssi_tensor, position


    def __getitem__(self, item):
        rssi_sample = self.val_set[item]
        position = rssi_sample[:self.dim_coord].astype(float)
        rssi_array = rssi_sample[-self.dim_rssi:].astype(float)

        position = torch.tensor(position, dtype=torch.float32)
        rssi_tensor = torch.tensor(rssi_array, dtype=torch.float32)

        return rssi_tensor, position

def get_dataloader_ntuiotrssi(path_dataset, batch_size):
    ntuiotrssi_dataset = NTUIoTRSSI_Dataset(path_dataset)
    ntuiotrssi_dataloader = DataLoader(dataset=ntuiotrssi_dataset, batch_size=batch_size, shuffle=True)
    return ntuiotrssi_dataloader


# if __name__ == "__main__":
#     ntuiot_dataset = NTUIoTRSSI_Dataset('path/to/your/data')
#     ntuiot_dataset.get_compact()
#     print(ntuiot_dataset.__getitem__(0))