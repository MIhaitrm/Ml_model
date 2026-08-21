#!/usr/bin/env python
import torch
import torch.nn.functional as F
import subprocess
import torch.nn as nn
from torch.nn import MSELoss
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pdb
from dummy_data import append_label
from graph_loader import append_pos_label, train_loader, train_dataset, test_dataset
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import random

dataset = train_dataset
try:
	subprocess.check_output('nvidia-smi')
except e:
	print('No GPU found!')

print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

# x = torch.Tensor([[1, 2],
# 	[3, 4]])
x = torch.rand(112, 112)
edge_index = torch.randint(1, 100, (2, 200))
y = torch.ones(len(x))
# dataset = Data(x=x, edge_index=edge_index, y=y)
# relu = torch.nn.ReLU()
# # out=r(x)
# # print(out)
# # g = out.pow(2).sum(1)
# # print("G:", g)
# def goodness(data):
# 	goodness = data.pow(2).sum(1)
# 	return goodness
# # x = goodness(x)
# # print(x)
# def loss_ff(out, positive):
# 	threshold = 0.5
# 	theta = threshold if positive else -threshold
# 	out = -x if positive else x  ### Loss is calculated different for positive and negative examples
# 	# print(out+theta)
# 	loss = torch.log(1+torch.exp(out+theta)).mean()
# 	# print("Loss: ",loss)
# 	return loss
# 	# loss = out - threshold
# 	# # print(loss)
# 	# loss = -loss if positive else loss
# 	# loss = torch.log(1 + torch.exp(loss)).mean()
# 	# print(loss)
# 	return loss

# # loss = loss_ff(g, True)
# # print(loss)

# def train_ff(data, positive):
# 		loss = loss_ff
# 		x = data
# 		# print(x)
# 		x = relu(x)
# 		x = goodness(x)
# 		print("Goodness: ", x)
# 		x = loss(x, positive)
# 		# optimizer.zero_grad()
# 		# loss.backward(retain_graph=True)
# 		# optimizer.step()
# 		print("Loss: ",x)
# 		return x

# train_ff(x, True)
# x = torch.Tensor([0, 1, 2, 3, 2, 1])
# x = torch.tensor(x, dtype=torch.long)
# print(F.one_hot(x))
# F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
# F.one_hot(torch.arange(0, 6).view(3,2) % 3)
random.seed(42)

def goodness(data):
	goodness = data.pow(2).mean(1)
	return goodness


def loss_ff(x, positive):
	threshold = 2
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
		x = data.x
		edge_index = data.edge_index
		# for d in dataset:
		# 	x, edge_index = data.x, data.edge_index
		# pdb.set_trace()
		g = []
		for layer in self.layers:
			x = x.detach()
			x = layer(x, edge_index)
			# pdb.set_trace()
			x = torch.nan_to_num(x)
			x = self.relu(x)
			g = goodness(x)
			g += g
			# print("G:", g)
			# pdb.set_trace()
			# print("G stack: ", torch.stack(tuple(g), 0).sum(0))
		return x, torch.stack(tuple(g), 0).sum(0)

	def train_ff(self, data, positive, optimizer):
		# for d in data:
		x, edge_index = data.x, data.edge_index
		# print(len(x))
		for layer in self.layers:
			x = x.detach()
			x = layer(x, edge_index)
			# print("X after layer:",x)
			x = self.relu(x)
		# 	print("X after relu:", x)
			out = goodness(x)
		# 	print("Goodness: ",x)
			loss = loss_ff(out, positive)
			loss = torch.tensor(loss, requires_grad=True) ### maybe False
		# 	# loss[loss=="inf"].data = torch.tensor([0], dtype=torch.float)
		# 	# print("Loss: ", loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# pdb.set_trace()
		# 	# pdb.set_trace()
		return loss


device='cpu'
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = GNN_FF().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train(model, train_loader, optimizer, device):
    model.train()
    # x = train_loader
    for epoch in range(10):
    	for graph in train_loader:
    		for i in range(2):
	    		if i == 0:
	    			label = torch.Tensor([0, 1])
	    		x = append_label(graph.x, graph.edge_index, label)
	    		# print("X pos values: ", x)
		    	loss = model.train_ff(x, False, optimizer)
		    	# print("Loss neg: ", loss)
		    	if i == 1:
		    		label = torch.Tensor([1, 0])
		    	x = append_label(graph.x, graph.edge_index, label)
		    	# print("X neg values: ", x)
		    	loss = model.train_ff(x, True, optimizer)
		    	# print("Loss pos: ", loss)

train(model=model, train_loader=train_dataset, optimizer=optimizer, device=device)

def predict(model, pred_data):
	# print("Len train data:", len(pred_data))
	out = pred_data
	g_for_label = []
	gg = []
	for sample in out:
		# print("Sample len:", sample.x.shape)
		for i in range(2):
			if i == 0:
				label = torch.Tensor([0,1])
			else:
				label = torch.Tensor([1,0])
			x = append_label(sample.x, sample.edge_index, label)
			# print("X:", x.shape)
			_, g = model.forward(x)
			gg += [g]
			# print("g:", g)
		good = torch.Tensor(gg).argmax(0)
		print("good:", good)
	g_for_label += [good]
	print("G_for_label:", g_for_label)
	# pdb.set_trace()
		# value = torch.Tensor(g_for_label).argmax(0)
		# print("Value: ", value)
	# print("argmax: ", torch.stack(g_for_label, 0).argmax(0))
	# print("Total goodness: ", torch.stack(g_for_label, 0))#.argmax(0))
	# return torch.stack(g_for_label, 0).argmax(0)
	return torch.stack(g_for_label, 0)
	# pdb.set_trace()

# predict(model=model, pred_data=train_dataset)

def test(model, test_loader, device):
    correct = 0
    total = 0
    pred_s = []
    corr = []
    model.eval()
    with torch.no_grad():
        for graph in test_loader:
            x, label = graph.x, graph.label
            pred = predict(model, test_loader)
            print("Prediction: ", pred)
            correct += (pred == label).sum().item()
            print("Correct: ", correct)
            total += label
            print("Total: ", total)
            acc = correct / total
            print("Accuracy: ", acc)
            pred_s.append(pred)
            corr.append(label)
            print("Pred_s: ",pred_s)
            print("Corr: ", corr)
            preds = []
            for i in pred_s:
            	if i == 0:
            		score = [0, 1]
            	else:
            		score = [1, 0]
            	preds += score
            print("Preds: ", preds)
            corrects = []
            for i in corr:
            	if i ==1:
            		s = [1, 0]
            	corrects += s
        mcc = matthews_corrcoef(torch.tensor(corrects), preds)
        print("MCC: ", mcc)
        print("Scores: ", classification_report(corr, pred_s, labels=[1]))
    return acc

test(model=model, test_loader=test_dataset, device=device)


