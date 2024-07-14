from torch_geometric.data import Dataset

from SynMapper.model.utils import process_data


class SynMapperDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.filename = filename
        self.transform = transform
        self.pre_transform = pre_transform
        self.data = process_data(filename)
        super(SynMapperDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
