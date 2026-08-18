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
from graph_loader import train_dataset

dataset_dir = "npy_files"
# dataset_dir = "npy_file"
# dataset_dir = "n_file"
# files = glob.glob(os.path.join(dataset_dir, "*.npy"))
# print()

# def negative_data(data):
#     for a in data:
#         x, edge_index, y = a.x, a.edge_index, a.y
#     x = torch.rand(len(x[:,0]), len(x[0,:]))
#     edge_index = torch.randint(low=min(edge_index[0]), high=max(edge_index[0]), size=(len(edge_index[:,0]), len(edge_index[1])))
#     # y = torch.zeros(len(y))
#     y = torch.Tensor([0])
#     return Data(x=x, edge_index=edge_index, y=y)


# class NEGDataset(Dataset):
#     def __init__(self, annotation_file, dataset_dir):
#         self.data = pd.read_csv(annotation_file)
#         print(f"CSV shape: {self.data.shape}")
#         self.dataset_dir = dataset_dir
#         print(len(self.dataset_dir))
#         print(isinstance(self.data, pd.DataFrame))
#         # self.label = torch.ones(len(self.dataset_dir))
#         # self.label = torch.Tensor([1])

#     def __len__(self):
#         return len(self.dataset_dir)

#     def __getitem__(self, index):
#         file_path = os.path.join(dataset_dir, self.data.iloc[index, 0])
#         file = torch.load(file_path, weights_only=False)
#         x = file['x']
#         edge_index = file['edge_index']
#         edge_index = edge_index.T
#         # x = torch.rand(len(x[:,0]), len(x[0,:]))
#         # edge_index = torch.randint(low=min(edge_index[0]), high=max(edge_index[0]), size=(len(edge_index[:,0]), len(edge_index[1])))
#         # y = torch.zeros(len(x))
#         # label = y.unsqueeze(1)
#         # x_long = torch.tensor(x, dtype=torch.long)
#         # label = torch.nn.functional.one_hot(x_long).float()
#         # x = torch.cat([x, label], dim=1)
#         dataset = Data(x=x, edge_index=edge_index, num_nodes=len(x))
#         dataset.label = 0
#         return dataset

# neg_data = NEGDataset(annotation_file="a_file.csv", dataset_dir=dataset_dir)

# # def create_negative_data():
# #     return torch.randint(0, 1, (1,))

# def append_neg_label(neg_data):
#     for n in neg_data:
#         x, edge_index, y = n.x, n.edge_index, n.label
#         label = torch.zeros(len(x))
#         label = label.unsqueeze(1)
#         x = torch.cat([x, label], dim=1)
#     return Data(x, edge_index)

# # app = append_neg_label(neg_data)
# # print(app.x.shape)
# # y = create_negative_data()
# # print(y)
# # append_labe(neg_data, y)
# # # pdb.set_trace()


# train_dataloader = DataLoader(neg_data, batch_size=64, shuffle=True)
# train_features = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")

# for n in neg_data:
#     x, edge_index = n.x, n.edge_index
#     print(x[0])
#     print(n.label)
    # print(len(n))
    # pdb.set_trace()



class NEGDataset(Dataset):
    def __init__(self, annotation_file, dataset_dir):
        self.data = pd.read_csv(annotation_file)
        print(f"CSV shape: {self.data.shape}")
        self.dataset_dir = dataset_dir
        print(isinstance(self.data, pd.DataFrame))
        # self.label = torch.Tensor([1]) ### One label for each graph, the better solution

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, index):
        file_path = os.path.join(dataset_dir, self.data.iloc[index, 0])
        file = torch.load(file_path, weights_only=False)
        x = file['x']
        edge_index = file['edge_index']
        edge_index = edge_index.T
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # y = torch.ones(len(x)) ### Label for each vertex, temporary solution for random splitting error
        # y = y.unsqueeze(1)
        # x = torch.cat([x, y], dim=1)
        dataset = Data(x=x, edge_index=edge_index, num_nodes=len(x))
        dataset.label= torch.tensor([0], dtype=torch.long)
        return dataset
        

neg_data = NEGDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
print("Num Features: ",neg_data.num_node_features)
print(neg_data.num_classes)

def append_neg_label(neg_data):
    for n in neg_data:
        x, edge_index, y = n.x, n.edge_index, n.label
        label_one = torch.ones(len(x))
        label_zero = torch.zeros(len(x))
        label_one = label_one.unsqueeze(1)
        label_zero = label_zero.unsqueeze(1)
        x = torch.cat([x, label_zero, label_one], dim=1)
    return Data(x, edge_index)


# def append_label(data):
#     for n in data:
#         x, edge_index, label = n.x, n.edge_index, n.label
#         label = label.repeat(len(x))
#         print(label.shape)
#         for i in range(2):
#             i = torch.tensor(i, dtype=torch.float)
#             label = i.repeat(len(x))
#             label = label.unsqueeze(1)
#             x = torch.cat([x, label], dim=1)
#             print(x[0])
#     return Data(x=x, edge_index=edge_index)

def append_label(data, edge_index, y):
    # for n in data:
    #     x, edge_index = n.x, n.edge_index
    # pdb.set_trace()
    y = y.repeat(len(data), 1)
    # print("Y shape:", y.shape)
    # print("X shape: ", data.x.shape)
    # y = y.unsqueeze(1)
    # y = list([])
    x = torch.cat([data, y], dim=1)
    # return x
    return Data(x=x, edge_index=edge_index)

# for sample in train_dataset:
#     # print(sample.x.shape)
#     for i in range(2):
#         if i == 0:
#             label = torch.Tensor([0,1])
#         else:
#             label = torch.Tensor([1,0])
#         append_label(sample.x, label)


# for i in range(2):
#     d = append_label(neg_data, torch.Tensor([i]))
#     print(d.x)

# x = torch.tensor([1])
# x = x.repeat(10)
# print(x)

# for i in range(1):
#     i = torch.tensor(i, dtype=torch.float)
#     label = i.repeat(100)
#     label.unsqueeze(1)
#     print(label)