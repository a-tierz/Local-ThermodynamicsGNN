import os
import torch
from glob import glob
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class GraphDataset_(Dataset):
    def __init__(self,  dInfo, dset_dir):
        self.folder = dset_dir
        self.files = glob(os.path.join(dset_dir, "*.pt"))
        self.files.sort()
        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {
            'z': self.z_dim,
            'q': dInfo['dataset']['q_dim'],
            'q_0': dInfo['dataset']['q0_dim'],
            'n': 1,
            'f': dInfo['dataset']['external_force_dim'],
            'g':dInfo['dataset']['g_dim'],
        }

        self.dt = dInfo['dataset']['dt']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False, map_location='cpu', mmap=True)
        return graph

    def get_stats(self):
           # Calculate scaling statistics
        all_y = []
        all_vel = []
        for f in self.files:
            g = torch.load(f, weights_only=False)
            all_y.append(g.y)
            if self.dims['f'] > 0:
                all_vel.append(g.vel)

        total = torch.cat(all_y, dim=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(total)

        if self.dims['f'] > 0:
            scaler_f = MinMaxScaler(feature_range=(-1, 1))
            scaler_f.fit(all_vel)
        else:
            scaler_f = None

        return scaler, scaler_f
        
class GraphDataset(Dataset):
    """
    Dataset class for loading graph data.

    Args:
        dInfo (dict): Information about the dataset.
        dset_dir (str): Directory where the dataset is located.
        length (int, optional): Length of the dataset to load. Defaults to 0.
    """

    def __init__(self, dInfo, dset_dir, length=0):
        self.dset_dir = dset_dir
        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {
            'z': self.z_dim,
            'q': dInfo['dataset']['q_dim'],
            'q_0': dInfo['dataset']['q0_dim'],
            'n': 1,
            'f': dInfo['dataset']['external_force_dim'],
            'g':dInfo['dataset']['g_dim'],
        }

        self.dt = dInfo['dataset']['dt']
        self.data = torch.load(dset_dir, weights_only=False)
        if length != 0:
            self.data = self.data[:length]

    def __getitem__(self, index):
        # Load data
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)

    def get_stats(self):
        """
        Compute statistics of the dataset.

        Returns:
            MinMaxScaler: Scaler object fitted to the dataset.
        """
        # total_tensor = torch.cat([data.x for data in self.data], dim=0)
        total_tensor = torch.cat([data.y for data in self.data], dim=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(total_tensor)

        if self.dims['f'] == 0:
            scaler_f = None
        else:
            total_tensor_f = torch.cat([data.f for data in self.data], dim=0)
            # total_tensor_f = torch.cat([data.vel for data in self.data], dim=0)
            scaler_f = MinMaxScaler(feature_range=(-1, 1))
            scaler_f.fit(total_tensor_f)
            scaler_f.min_[0] = 0

        return scaler, scaler_f


if __name__ == '__main__':
    pass
