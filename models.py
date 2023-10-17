import sys, os
# sys.path.append(os.path.abspath(os.path.join('../..')))

import torch
import numpy as np
import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
from utils.data_load import read_dataset

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
import sys
import random
import argparse
import logging
import time

class NetRegression(nn.Module):
    def __init__(self, input_size, num_classes, size = 50):
        super(NetRegression, self).__init__()
        self.first = nn.Linear(input_size, size)
        self.fc = nn.Linear(size, size)
        self.last = nn.Linear(size, num_classes)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.fc(out))
        out = self.last(out)
        return self.sigmoid(out)

class Adversary(nn.Module):
    
    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

class NeuralGBDT(nn.Module):
    def __init__(self, input_dim, output_dim, n_ensemble, hidden_dim=50):
        super(NeuralGBDT, self).__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            ) for _ in range(n_ensemble)
        ])

    def forward(self, x):
        # Mimic the additive behavior of boosting
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# GBDT Wrapper that behaves like a PyTorch model
class GBDTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(GBDTWrapper, self).__init__()
        self.model = GradientBoostingRegressor(**kwargs)
        
    def fit(self, X, y):
        # Move data to CPU for sklearn model
        self.model.fit(X.cpu().detach().numpy(), y.cpu().detach().numpy())
        
    def forward(self, X):
        # Predict using the model and move output back to the GPU (or whichever device X is on)
        return torch.tensor(self.model.predict(X.cpu().detach().numpy())).to(X.device)





