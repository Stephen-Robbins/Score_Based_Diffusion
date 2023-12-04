import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import diffusion
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch
from model import PositionalEmbedding

    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120+32, 84)
        self.fc3 = nn.Linear(84, 10)

        ## Time embedding
        self.time_mlp = PositionalEmbedding(32)

    def forward(self, x, t):
        embed = self.time_mlp(t)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(torch.concat([self.fc1(x), embed], axis = 1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)

def cross_entropy_loss_function(class_net, x, y, sde, eps=1e-5):
    '''
    Cross entropy loss for conditional generation of MNIST
    '''
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 
    z = torch.randn_like(x, device = x.device)
    mu, std= sde.marginal(x, random_t)
    perturbed_x=mu+std*z
    y_hat = class_net(perturbed_x, random_t)
    '''
    Lambda: retrieves appropriate lambda based off of SDE
    '''
    loss = torch.nn.functional.cross_entropy(y_hat, y)
    
    return loss

def train_classification_network_mnist(dataloader, class_net, sde, optimizer, epochs=100):
    avg=0
    device = sde.device

    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = cross_entropy_loss_function(class_net, x, y, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(class_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        print(f'Epoch: {epoch} and Loss: {avg}' )
        avg=0

        torch.save(class_net.state_dict(), f'./models/MNISTClassifier/epoch{epoch}')