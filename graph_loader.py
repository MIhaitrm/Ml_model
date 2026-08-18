#!/usr/bin/env python

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import glob
import os
import pandas as pd
from torch_geometric.datasets import TUDataset
import pdb
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
        dataset.label=1
        return dataset
        

# data_pos = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=dataset_dir)
# # print("Num Features:",data)
# # pdb.set_trace()
# print(len(data_pos))
# for sample in data_pos:
#     x = sample.x
#     print("Len graph:", x.shape)

def append_pos_label(data):
    for n in data:
        x, edge_index, y = n.x, n.edge_index, n.label
        label_one = torch.ones(len(x))
        label_zero = torch.zeros(len(x))
        label_one = label_one.unsqueeze(1)
        label_zero = label_zero.unsqueeze(1)
        x = torch.cat([x, label_one, label_zero], dim=1)
    return Data(x, edge_index)


content = os.listdir(dataset_dir)
files = np.array(content)
print(files)

train_set, test_set = train_test_split(files, test_size=0.2)
print("train_set",train_set)


train_dataset = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=train_set)
test_dataset = RNNDataset(annotation_file="annotation_file.csv", dataset_dir=test_set)

# for i in train_dataset:
#     print("Graph shape: ", i.x.shape)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print("test_set: ", test_set)

# def negative_labels():
# 	return torch.zeros(1)

# def encode_label_in_graph(x, y):
# 	# y = torch.nn.functional.one_hot(y, 10).float()
# 	# label = torch.zeros(10)
# 	x[10:,:] = y
# 	return x

# for d in data:
# 	x, edge_index, y = d.x, d.edge_index, d.y
# 	x = torch.tensor(x, dtype=torch.long)
# 	print(x[10,:])
# 	x = x[10:, :]
# 	neg_graph = encode_label_in_graph(x, torch.zeros(10, dtype=torch.long))
# 	# print(neg_graph)
# 	# pdb.set_trace()