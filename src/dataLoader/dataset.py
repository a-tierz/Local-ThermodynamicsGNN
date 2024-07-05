import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


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
        self.y_dim = len(dInfo['dataset']['state_variables_out'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {
            'z': self.z_dim,
            'y': self.y_dim,
            'q': dInfo['dataset']['q_dim'],
            'q_0': dInfo['dataset']['q0_dim'],
            'n': 1,
            'f': dInfo['dataset']['external_force_dim'],
            'g':dInfo['dataset']['g_dim'],
        }

        self.dt = dInfo['dataset']['dt']
        self.data = torch.load(dset_dir)
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
        total_tensor = torch.cat([data.x for data in self.data], dim=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(total_tensor)
        return scaler

    def get_stats2(self):
        """
        Compute statistics of the dataset.

        Returns:
            MinMaxScaler: Scaler object fitted to the dataset.
        """
        total_tensor = torch.cat([data.x for data in self.data], dim=0)
        scaler_pu = MinMaxScaler(feature_range=(0, 1))
        scaler_pu.fit(total_tensor)

        total_tensor = torch.cat([data.y for data in self.data], dim=0)
        scaler_s = MinMaxScaler(feature_range=(0, 1))
        scaler_s.fit(total_tensor)

        return scaler_pu, scaler_s


if __name__ == '__main__':
    pass
