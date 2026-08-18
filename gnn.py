#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch_geometric.nn import GATConv, GCNConv
# from graph_loader import loader
import numpy as np
# dataset = loader
threshold = 0.5
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
from load_npy import train_dataloader
from load_npy import data
from torch_geometric.transforms import RandomNodeSplit
import pdb
from negative_data import neg_data

dataset = data
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# pdb.set_trace()
# print(dataset)
# print(dataset.num_classes)
def goodness(data):
	goodness = data.pow(2).sum(1)
	return goodness

threshold=0.0
class GNN_FF(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = GCNConv(dataset.num_node_features, dataset.num_classes)
		# self.layer2 = GCNConv(32, dataset.num_classes)
		self.relu = torch.nn.ReLU()
		self.threshold = threshold
		self.goodness = goodness
		# self.layers = [self.layer1, self.layer2]
	
	def forward(self, data):
		# x, edge_index = data.x, data.edge_index
		for d in data:
			x, edge_index, y = d.x, d.edge_index, d.y
		# for e in data:
		# 	edge_index = e.edge_index
		# pdb.set_trace()
		# x = self.layer1(x, edge_index)
		# x = self.relu(x)
		# # print("X after relu: ", min(x[0]))
		# x = self.goodness(x)
		# x = F.dropout(x, training=self.training)
		# x = self.layer2(x, edge_index)
		# x = self.relu(x)
		# print("X after relu in layer2: ", x)
		# x = self.goodness(x)
		# print("x after goodness: ", x)
		# for layer in self.layers:
		# 	x = layer(x, edge_index)
		# 	x = self.relu(x)
		# 	print("X after relu: ", x)
		# 	x = self.goodness(x)
		# return x	
		x = self.layer1(x, edge_index)
		x = self.relu(x)
		x = self.goodness(x)
		
		return x


	
	# def loss(self, x, positive):
	# 	# x, edge_index = data.x, data.edge_index
	# 	# for layer in self.layers:
	# 		# out = layer(x)
	# 		# out = self.relu(out)
	# 		# out = self.goodness(out)
	# 	loss = x - self.threshold
	# 	loss = -loss if positive else loss
	# 	loss = torch.log(1+torch.exp(loss)).mean()
	# 	print("Loss: ",loss)
	# 	return loss
node_transform = RandomNodeSplit(num_val=0.2, num_test=0.3)
for da in data:
	node_splits = node_transform(da)

def loss_ff(x, positive):
		# x, edge_index = data.x, data.edge_index
		# for layer in self.layers:
			# out = layer(x)
			# out = self.relu(out)
			# out = self.goodness(out)
		threshold = 0.5
		# g = goodness(x)
		loss = x - threshold
		loss = -loss if positive else loss
		loss = torch.log(1+torch.exp(loss)).mean()
		print("Loss: ",loss)
		return loss

# def negative_data(data):
# 	for a in data:
# 		x, edge_index, y = a.x, a.edge_index, a.y
# 	x = torch.rand(len(x[:,0]), len(x[0,:]))
# 	edge_index = torch.randint(low=min(edge_index[0]), high=max(edge_index[0]), size=(len(edge_index[:,0]), len(edge_index[1])))
# 	# y = torch.zeros(len(y))
# 	y = torch.Tensor([0])
# 	return Data(x=x, edge_index=edge_index, y=y)
	
device='cpu'
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = GNN_FF().to(device)
# data = dataset[0].to(device)
data = dataset
print(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# loss = loss(data, True)

def train(model, train_loader, neg_loader, optimizer, device):
    model.train()
    for epoch in range(10):
    	optimizer.zero_grad()
    	out = model(data)
    	loss = loss_ff(out, True)
    	# loss = MSELoss()
    	# loss = F.nll_loss(out[node_splits.train_mask], data.y[node_splits.train_mask])
    	# x_neg = negative_data(neg_data)
    	# pdb.set_trace()
    	# print(x_neg)
    	# x_neg = model(neg_data)
    	# loss = loss_ff(x_neg, False)
    	loss.backward()
    	optimizer.step()
    	# pdb.set_trace()

train(model=model, train_loader=data, neg_loader=neg_data, optimizer=optimizer, device=device)


node_transform = RandomNodeSplit(num_val=0.2, num_test=0.3)
for da in data:
	node_splits = node_transform(da)
	print("Pos splits: ", node_splits.test_mask)
	# print(node_splits.val_mask)

for n in neg_data:
	neg_node_splits = node_transform(n)
	print("Neg splits:", neg_node_splits.test_mask)
	# print(node_splits.val_mask)

# ys = []
for item in neg_data:
	y = item.y
	# print("Y: ", y)

# for item in data:
# 	y = item.y
# 	print("Y:", y)
	# print(0 in y)
model.eval()
pred = model(neg_data)
# print("Y neg: ", y[neg_node_splits.test_mask])
# print(y[neg_node_splits.test_mask]).sum()
correct = (pred[neg_node_splits.test_mask] == y[neg_node_splits.test_mask]).sum()
print(correct)
acc = int(correct) / int(neg_node_splits.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# pred = model(data)
# print("Pred: ",pred)
# # # print("Y neg: ", y[neg_node_splits.test_mask])
# # # print(y[neg_node_splits.test_mask]).sum()
# correct = (pred[node_splits.test_mask] == y[node_splits.test_mask]).sum()
# print(correct)
# acc = int(correct) / int(node_splits.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')