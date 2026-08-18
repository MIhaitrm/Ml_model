#!/usr/bin/env python


import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
# from load_npy import train_dataloader
# from load_npy import data
from graph_loader import train_loader, test_loader, append_pos_label, data
from dummy_data import append_neg_label, append_label
from torch_geometric.transforms import RandomNodeSplit
import pdb
# from negative_data import neg_data

# dataset = data
dataset=data

### Define the goodness for each layer as the sum of squares of relu
def goodness(data):
	goodness = data.pow(2).sum(1)
	return goodness


def loss_ff(x, positive):
	threshold = 0.5
	theta = threshold if positive else -threshold
	out = -x if positive else x  ### Loss is calculated different for positive and negative examples
	loss = torch.log(1+torch.exp(out+theta)).mean()
	# print("Loss: ",loss)
	return loss


class GNN_FF(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = GCNConv(-1, 32)
		self.layer2 = GCNConv(32, 2)
		self.norm = torch.nn.LayerNorm(dataset.num_node_features, 32)
		self.relu = torch.nn.ReLU()
		self.goodness = goodness
		self.loss_ff = loss_ff
		self.layers = [self.layer1, self.layer2]
	
	def forward(self, data):
		for d in data:
			x, edge_index, y = d.x, d.edge_index, d.y
		goodness = []
		for layer in self.layers:
			x = layer(x, edge_index)
			x = self.relu(x)
			g = goodness(x)
			goodness += x.pow(x).mean(1)
		return x, torch.stack(goodness, 1).sum(1)

	def train_ff(self, data, positive, optimizer):
		x, edge_index = data.x, data.edge_index
		# print(x)
		for layer in self.layers:
			x = x.detach()
			x = layer(x, edge_index)
			print("X after layer:",x)
			x = self.relu(x)
			# print("X after relu:", x)
			x = goodness(x)
			print("Goodness: ",x)
			loss = loss_ff(x, positive)
			loss = torch.tensor(loss, requires_grad=True)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# pdb.set_trace()
		return loss


device='cpu'
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = GNN_FF().to(device)
# data = dataset
# print(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


### Train on 2 different datasets, perhaps better to concatenate the datasets
### Initializing the negative examples as random values is not be the best method
### I think I should give negative labels to some positive examples and 
### the algorithm should learn to distinguish between them? Hinton approach from I what I understood
def train(model, train_loader, optimizer, device):
    model.train()
    for epoch in range(10):
    	x = append_pos_label(train_loader)
    	loss = model.train_ff(x, True, optimizer)
    	x = append_neg_label(train_loader)
    	loss = model.train_ff(x, False, optimizer)


train(model=model, train_loader=train_loader, optimizer=optimizer, device=device)


# node_transform = RandomNodeSplit(num_val=0.2, num_test=0.3)
# for da in data:
# 	node_splits = node_transform(da)
# 	print("Pos splits: ", node_splits.test_mask)

# for n in neg_data:
# 	neg_node_splits = node_transform(n)
# 	print("Neg splits:", neg_node_splits.test_mask)

# for item in neg_data:
# 	y = item.y

# # for item in data:
# # 	y = item.y

# model.eval()
# pred = model(neg_data)
# # print("Y neg: ", y[neg_node_splits.test_mask])
# # print(y[neg_node_splits.test_mask]).sum()
# correct = (pred[neg_node_splits.test_mask] == y[neg_node_splits.test_mask]).sum()
# print(correct)
# acc = int(correct) / int(neg_node_splits.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')

# pred = model(data)
# print("Pred: ",pred)
# correct = (pred[node_splits.test_mask] == y[node_splits.test_mask]).sum()
# print(correct)
# acc = int(correct) / int(node_splits.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')