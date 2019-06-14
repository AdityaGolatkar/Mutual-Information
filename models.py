import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from utils import *

class Network(nn.Module):
    
    def __init__(self,dropout=False,hidden_size=512,input_size=784):
        super(Network, self).__init__()
        
        self.features = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(inplace=True, p=.5) if dropout else Identity(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(inplace=True, p=.5) if dropout else Identity(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        #nn.Dropout(inplace=True, p=.5) if dropout else Identity(),
        )
        
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, 10)
        )
        
    def forward(self, x):
        features = self.reparameterize(self.features(x))
        output = self.classifier(features)
        return output,features
        
    def reparameterize(self, mu):
        std = 0.1
        eps = torch.randn_like(mu)
        return mu + eps*std
    
class Statistic_Network(nn.Module):
    def __init__(self, hidden_size=512,input_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = torch.tanh(output)
        return output