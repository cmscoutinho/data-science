# Importação de pacotes
import torch
from torch import nn, optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

import pandas as pd

# Definição dos hiperparâmetros
args = {'num_epochs': 30,
		'batch_size': 30,
		'lr': 1e-4,
		'weight_decay': 5e-4,
		'num_workers': 1}

# Verificação de disponbilidade da GPU
if torch.cuda.is_available():
	args['device'] = torch.device('cuda')
else:
	args['device'] = torch.device('cpu')

print(args)

df = pd.read_csv('hour.csv')
print(len(df))
print(df.head())

indices = torch.randperm(len(df)).tolist()
train_size = int(0.8*len(df))
df_train   = df.iloc[indices[:train_size]]
df_test    = df.iloc[indices[train_size:]]

df_train.to_csv('bike_train.csv')
df_test.to_csv('bike_test.csv')

class Bicicletas(Dataset):
	def __init__(self, csv_path):
		self.dados = pd.read_csv(csv_path).to_numpy()

	def __getitem__(self, idx):
		sample = self.dados[idx][2:14]
		label  = self.dados[idx][-1:]

		# Conversão para tensor
		sample = torch.from_numpy(sample.astype(np.float32))
		label  = torch.from_numpy(label.astype(np.float32))

		return sample, label

	def __len__(self):
		return len(self.dados)

train_set = Bicicletas('bike_train.csv')
test_set = Bicicletas('bike_test.csv')

train_loader = DataLoader(train_set,
						  batch_size=args['batch_size'],
						  shuffle=True,
						  num_workers=args['num_workers'])

test_loader = DataLoader(test_set,
						  batch_size=args['batch_size'],
						  shuffle=True,
						  num_workers=args['num_workers'])

class MLP(nn.Module):
	def __init__(self, in_size, hid_size, out_size):
		super(MLP, self).__init__()

		self.features = nn.Sequential(nn.Linear(in_size, hid_size),
									  nn.ReLU(),
									  nn.Linear(hid_size, hid_size),
									  nn.ReLU())

		self.out = nn.Linear(hid_size, out_size)

	def forward(self,X):

		features = self.features(X)
		output   = self.out(features)

		return output

in_size  = 12
hid_size = 128
out_size = 1

net = MLP(in_size, hid_size, out_size).to(args['device'])
criterion = nn.L1Loss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

def train(epoch, net, train_loader):
	epoch_loss = []
	
	for batch in train_loader:
		dado, rotulo = batch

		pred = net(dado)
		loss = criterion(pred,rotulo)
		epoch_loss.append(loss.cpu())

		loss.backward()
		optimizer.step()

	epoch_loss = np.asarray(epoch_loss)
	print('Época %d. Loss: %.2f.' % (epoch, epoch_loss.mean()))

def test():
	pass

if __name__ == '__main__':
	for epoch in range(args['num_epochs']):
		train(epoch, net, train_loader)