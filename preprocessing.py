#!/usr/bin/env python


import pandas as pd
import os
import torch
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset, DataLoader
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

dataset_dir = "npy_files"
class RNNDataset(Dataset):
    def __init__(self, annotation_file, dataset_dir):
        self.data = pd.read_csv(annotation_file)
        print(f"CSV shape: {self.data.shape}")
        self.dataset_dir = dataset_dir
        print(isinstance(self.dataset_dir, np.ndarray))

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, index):
        file_path = os.path.join(dataset_dir, self.data.iloc[index, 0])
        file = np.load(file_path, allow_pickle=True)
        x = file['x']
        edge_index = file['edge_index']
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.float32)
        self.label = 1
        dataset = Data(x=x, edge_index=edge_index, y=self.label)
        return dataset

data = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
print(data)
# dataloader = DataLoader(data, shuffle=True, batch_size=64)
# train_features = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")


# node_transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
# data_splits = node_transform(data)

# dataloader = DataLoader(data_splits, shuffle=True, batch_size=64)