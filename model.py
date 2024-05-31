import numpy as np
import math
import torch
from torch import nn
from collections import OrderedDict


class WKNN:
    def __init__(self, database_position, database_rssi):
        super(WKNN, self).__init__()
        self.database_position = database_position
        self.database_rssi = database_rssi

    def compute_similarity(self, point_query, point_support):
        rssi_err = point_query - point_support
        abs_err = np.linalg.norm(rssi_err)

        abs_err += 1e-4 if abs_err == 0 else 0
        similarity = 1 / abs_err
        return similarity

    def __call__(self, input_rssi, K):
        self.num_best = K
        len_base = len(self.database_rssi)
        similarity = np.zeros(len_base)
        for i in range(0, len_base):
            similarity[i] = self.compute_similarity(input_rssi, self.database_rssi[i])
        idx_similarity = np.argsort(similarity, axis=-1, kind='quicksort', order=None)[::-1]
        similarity_ordered = similarity[idx_similarity]
        neighbor_position = self.database_position[idx_similarity[:self.num_best], :]
        neighbor_similarity = similarity[idx_similarity[:self.num_best]]
        neighbor_weight = neighbor_similarity / sum(neighbor_similarity)
        estimate_position = np.average(neighbor_position, weights=neighbor_weight, axis=0)
        return estimate_position
