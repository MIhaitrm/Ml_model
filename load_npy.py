#!/usr/bin/env python

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import glob
import os
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
import pdb

dataset_dir = "npy_files"

class RNNDataset(Dataset):
    def __init__(self, annotation_file, dataset_dir):
        self.data = pd.read_csv(annotation_file)
        print(f"CSV shape: {self.data.shape}")
        self.dataset_dir = dataset_dir
        print(isinstance(self.data, pd.DataFrame))
        # self.label = torch.ones(len(self.dataset_dir))
        # self.label = torch.Tensor([1])
        # print(self.label)

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, index):
        file_path = os.path.join(dataset_dir, self.data.iloc[index, 0])
        file = torch.load(file_path, weights_only=False)
        x = file['x']
        edge_index = file['edge_index']
        edge_index = edge_index.T
        x = torch.tensor(x, dtype=torch.float)
        # x = x.detach().clone().requires_grad_(True)
        # edge_index = edge_index.detach().clone()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # label = self.label
        label = torch.ones(len(x))
        label = torch.tensor(label, dtype=torch.long)
        dataset = Data(x=x, edge_index=edge_index, y=label, num_nodes=len(x))
        # node_transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
        # node_splits = node_transform(dataset)
        # return x, edge_index, label
        return dataset

# pdb.set_trace()
data = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
# pdb.set_trace()
print(data.__dict__)
print("Num classes: ",data.num_classes)
# data.x = []
for d in data:
	x = d.x
	print(isinstance(x, torch.Tensor))
# 	data.x.append(x)
# print(data.x)
# pdb.set_trace()
train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# pdb.set_trace()