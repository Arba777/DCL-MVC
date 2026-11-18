from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import math
import random

class SingleviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return data, mask


def get_mask(num_views, data_size, missing_rate):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask

class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    
def load_data(args):
    if args.dataset_name == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
        args.multiview_dims = dims
        args.num_views = view
        args.class_num = class_num
        args.data_size = data_size
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

def build_dataset(args):
    dataset, dims, num_views, data_size, class_num = load_data(args)

    data_list = [np.zeros((data_size, d), dtype=np.float32) for d in dims]
    labels = np.zeros(data_size, dtype=np.int32)

    for i in range(data_size):
        data, label, idx = dataset[i]
        labels[i] = label.numpy() if isinstance(label, torch.Tensor) else label
        for v in range(num_views):
            data_list[v][i] = data[v].numpy().reshape(-1)

    mask = get_mask(num_views, data_size, args.missing_rate)

    data_list_masked = [
        data_list[v] * mask[:, v:v+1]
        for v in range(num_views)
    ]

    Incomplete_dataset = Incomplete_MultiviewDataset(data_list_masked, mask, labels, num_views)

    complete_data = {
        'data' : data_list,
        'labels' : labels,
        'mask': mask
    }

    exist_singleView_data = [
        SingleviewDataset(data_list[v]) for v in range(num_views)
    ]
    return complete_data, Incomplete_dataset, exist_singleView_data
