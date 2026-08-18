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
from sklearn.model_selection import train_test_split
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
        dataset = Data(x=x, edge_index=edge_index, y=label, num_nodes=len(x))
        # node_transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
        # node_splits = node_transform(dataset)
        # return x, edge_index, label
        return dataset

# pdb.set_trace()
data = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
# pdb.set_trace()
print(data.__dict__)
# print(data.x)
# data.x = []
for d in data:
	x = d.x
	print(isinstance(x, torch.Tensor))
# 	data.x.append(x)
# print(data.x)
# pdb.set_trace()
# train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
# train_features = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")

def negative_data(data):
    for a in data:
        x, edge_index, y = a.x, a.edge_index, a.y
    x = torch.rand(len(x[:,0]), len(x[0,:]))
    edge_index = torch.randint(low=min(edge_index[0]), high=max(edge_index[0]), size=(len(edge_index[:,0]), len(edge_index[1])))
    # y = torch.zeros(len(y))
    y = torch.Tensor([0])
    return Data(x=x, edge_index=edge_index, y=y)

# neg = negative_data(data)
# print(neg.x.shape)
# pdb.set_trace()


class NEGDataset(Dataset):
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
        x = torch.rand(len(x[:,0]), len(x[0,:]))
        edge_index = torch.randint(low=min(edge_index[0]), high=max(edge_index[0]), size=(len(edge_index[:,0]), len(edge_index[1])))
        # y = torch.zeros(len(y))
        # y = torch.Tensor([0])
        # x = x.detach().clone().requires_grad_(True)
        # edge_index = edge_index.detach().clone()
        # edge_index = torch.tensor(edge_index, dtype=torch.long)
        # label = self.label
        label = torch.zeros(len(x))
        dataset = Data(x=x, edge_index=edge_index, y=label, num_nodes=len(x))
        # node_transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
        # node_splits = node_transform(dataset)
        # return x, edge_index, label
        return dataset


neg_data = NEGDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
train_dataloader = DataLoader(neg_data, batch_size=64, shuffle=True)
train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

content = os.listdir(dataset_dir)
files = np.array(content)
print(files)

train_set, test_set = train_test_split(files, test_size=0.2)
print("train_set",train_set)

neg_train_dataset = NEGDataset(annotation_file="annotation_file.csv", dataset_dir=train_set)
neg_test_dataset = NEGDataset(annotation_file="annotation_file.csv", dataset_dir=test_set)
