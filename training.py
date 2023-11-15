
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import diffusion
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config.yaml')
num_steps= config['num_steps']
num_samples=config['num_samples']
bmin=config['bmin']
bmax=config['bmax']
learning_rate=config['learning_rate']
epochs=config['epochs']

def loss_function(score_net, x, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 
    z = torch.randn_like(x)
    mu, std=diffusion.p(x, random_t)
    perturbed_x=mu+std*z
    score= score_net(perturbed_x, random_t)
    lamb=1/diffusion.B(random_t)
    lamb = lamb.unsqueeze(1)
    loss = torch.mean(lamb*torch.square((std*score  + z)))
    
    return loss

def train_score_network(dataloader, score_net, epochs=epochs):
    optimizer = optim.Adam(score_net.parameters(), lr=learning_rate)
    avg=0
    for epoch in range(epochs):
        for x_batch, in dataloader:
            optimizer.zero_grad()
            loss = loss_function(score_net, x_batch)
            loss.backward()
            #nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        if (epoch%100==0):
            print(f'Epoch: {epoch} and Loss: {avg/(8*100)}' )
            avg=0
        if(epoch%1000==0 and epoch!=0):
            samples = diffusion.backward_diffusion(score_net)
            data=x_batch.detach().numpy()
            samples_np = samples.detach().numpy()
            plt.scatter(data[:, 0], data[:, 1], label='Original Data')
            plt.scatter(samples_np[:, 0], samples_np[:, 1], label='Generated Samples')
            plt.legend()
            plt.show()