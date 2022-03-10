# TODO 09/03/2022 -> acrescentar fluxo de validação
# Importação de pacotes
import torch
from torch import nn, optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

import numpy as np

import time

# Definição dos argumentos do treinamento
args = {'batch_size': 30,
		'num_epochs': 30,
		'num_workers': 4,
		'lr': 1e-4,
		'weight_decay': 5e-4}

# Verificação de disponibilidade de GPU
if torch.cuda.is_available():
	args['device'] = torch.device('cuda')
else:
	args['device'] = torch.device('cpu')

# Definição da classe MLP
class MLPClassifier(nn.Module):
	def __init__(self, in_size, hid_size, out_size):
		super(MLPClassifier, self).__init__()

		self.features = nn.Sequential(nn.Linear(in_size, hid_size),
									  nn.ReLU(),
									  nn.Linear(hid_size, hid_size),
									  nn.ReLU())

		self.out      = nn.Linear(hid_size, out_size)
		self.softmax  = nn.Softmax()

	def forward(self, X):

		X = X.view(X.size(0), -1)

		features = self.features(X)
		output   = self.softmax(self.out(features))

		return output

# Carregamento do dataset
train_set = datasets.MNIST('./mnist',
							train=True,
							transform=transforms.ToTensor(),
							download=True)

test_set = datasets.MNIST('./mnist',
							train=False,
							transform=transforms.ToTensor(),
							download=False)

# Definição de hiperparâmetros
in_size  = len(train_set.data[0]) ** 2
hid_size = 128 
out_size = len(train_set.classes)

# Instanciação da rede, função de perda e otimizador
net = MLPClassifier(in_size, hid_size, out_size).to(args['device'])
criterion = nn.CrossEntropyLoss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

# Definição do dataloader
train_loader = DataLoader(train_set,
						  batch_size=args['batch_size'],
						  shuffle=True,
						  num_workers=args['num_workers'])

test_loader = DataLoader(test_set,
						 batch_size=args['batch_size'],
						 shuffle=True,
						 num_workers=args['num_workers'])

# Loop principal de treinamento
if __name__ == '__main__':

	# Iteração em cada época
	for epoch in range(args['num_epochs']):

		epoch_loss = []
		start = time.time()

		# Iteração em cada batch
		for batch in train_loader:

			dado, rotulo = batch

			dado = dado.to(args['device'])
			rotulo = rotulo.to(args['device'])

			# Forward
			pred = net(dado)
			loss = criterion(pred, rotulo)
			epoch_loss.append(loss.data.cpu())

			# Backward
			loss.backward()
			optimizer.step()
		
		end = time.time()
		time_elapsed = end-start
		
		epoch_loss = np.asarray(epoch_loss)
		
		print('Época: %d, loss: %.2f, +/- %.2f, time: %.2f.' % (epoch+1, epoch_loss.mean(), epoch_loss.std(), time_elapsed))
			