from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class COIL20(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'COIL20.mat')['Y'].astype(np.int32).reshape(1440, )
        self.V1 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 2].astype(np.float32)

    def __len__(self):
        return 1440

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(1024)
        x2 = self.V2[idx].reshape(3304)
        x3 = self.V3[idx].reshape(6750)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRCv1(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MSRCv1.mat')['Y'].astype(np.int32).reshape(210, )
        self.V1 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 4].astype(np.float32)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(24)
        x2 = self.V2[idx].reshape(576)
        x3 = self.V3[idx].reshape(512)
        x4 = self.V4[idx].reshape(256)
        x5 = self.V5[idx].reshape(254)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()




class DHA(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'DHA.mat')['Y'].astype(np.int32).reshape(483, )
        self.V1 = scipy.io.loadmat(path + 'DHA.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'DHA.mat')['X'][0, 1].astype(np.float32)


    def __len__(self):
        return 483

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(110)
        x2 = self.V2[idx].reshape(6144)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class nus_wide(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/nus-wide/L.mat')['L'].reshape(20000, )
        self.V1 = scipy.io.loadmat('./data/nus-wide/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/nus-wide/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class flickr(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/flickr30k/L.mat')['L'].reshape(12154, )
        self.V1 = scipy.io.loadmat('./data/flickr30k/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/flickr30k/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 12154

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class ESP_Game(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/esp-game/L.mat')['L'].reshape(11032, )
        self.V1 = scipy.io.loadmat('./data/esp-game/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/esp-game/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 11032

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 3304, 6750]
        view = 3
        data_size = 1440
        class_num = 20
    elif dataset == 'MSRCv1':
        dataset = MSRCv1('./data/')
        dims =[24,576,512,256,254]
        view = 5
        data_size = 210
        class_num = 7
    elif dataset == 'DHA':
        dataset = DHA('./data/')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == 'nus-wide':
        dataset = nus_wide('./data/')
        dims = [100, 100]
        view = 2
        data_size = 20000
        class_num = 8
    elif dataset == 'flickr':
        dataset = flickr('./data/')
        dims = [100,100]
        view = 2
        data_size = 12154
        class_num = 6

    elif dataset == 'ESP-Game':
        dataset = ESP_Game('./data/')
        dims = [100, 100]
        view = 2
        data_size = 11032
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
