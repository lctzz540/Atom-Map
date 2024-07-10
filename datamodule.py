import torch
from torch_geometric.loader import DataLoader
import numpy as np
from SynMapper.model.dataset import SynMapperDataset


class SynMapperDataModule:
    def __init__(self, file_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=420):
        assert sum([train_ratio, val_ratio, test_ratio]) == 1
        self.file_name = file_name
        self.dataset = self.load_dataset()
        self.num_examples = len(self.dataset)
        rng = np.random.default_rng(seed)
        self.shuffled_index = rng.permutation(self.num_examples)
        self.train_split = self.shuffled_index[:int(self.num_examples * train_ratio)]
        self.val_split = self.shuffled_index[
                         int(self.num_examples * train_ratio):int(self.num_examples * (train_ratio + val_ratio))]
        self.test_split = self.shuffled_index[int(self.num_examples * (train_ratio + val_ratio)):]

    def load_dataset(self, transform=None):
        return SynMapperDataset(self.file_name, transform=transform)

    def get_subset(self, indices):
        return [self.dataset[i] for i in indices]

    def loader(self, split, **loader_kwargs):
        subset = self.get_subset(split)
        return DataLoader(subset, **loader_kwargs)

    def train_loader(self, **loader_kwargs):
        return self.loader(self.train_split, shuffle=True, **loader_kwargs)

    def val_loader(self, **loader_kwargs):
        return self.loader(self.val_split, shuffle=False, **loader_kwargs)

    def test_loader(self, **loader_kwargs):
        return self.loader(self.test_split, shuffle=False, **loader_kwargs)
