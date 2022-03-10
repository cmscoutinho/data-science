 # Importação de pacotes
import torch
from torch import nn, optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

df = pd.read_csv