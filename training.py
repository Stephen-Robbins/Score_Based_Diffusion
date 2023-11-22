
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
# bmin=config['bmin']
# bmax=config['bmax']
learning_rate=config['learning_rate']
epochs=config['epochs']

def loss_function(score_net, x, sde, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 
    z = torch.randn_like(x, device = x.device)
    mu, std= sde.marginal(x, random_t)
    perturbed_x=mu+std*z
    score= score_net(perturbed_x, random_t)

    '''
    Lambda: retrieves appropriate lambda based off of SDE
    '''
    lamb=1/sde.B(random_t)
    lamb = lamb.unsqueeze(1)

    print(lamb.shape, std.shape, score.shape, z.shape)
    loss = torch.mean(lamb*torch.square((std*score  + z)))
    
    return loss

def train_score_network(dataloader, score_net, sde, epochs=epochs):
    optimizer = optim.Adam(score_net.parameters(), lr=learning_rate)
    avg=0
    for epoch in range(epochs):
        for x_batch, in dataloader:
            optimizer.zero_grad()
            loss = loss_function(score_net, x_batch, sde)
            loss.backward()
            #nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        if (epoch%100==0):
            print(f'Epoch: {epoch} and Loss: {avg/(8*100)}' )
            avg=0
        if(epoch%1000==0 and epoch!=0):
            samples = sde.backward_diffusion(score_net)
            data=x_batch.detach().numpy()
            samples_np = samples.detach().numpy()
            plt.scatter(data[:, 0], data[:, 1], label='Original Data')
            plt.scatter(samples_np[:, 0], samples_np[:, 1], label='Generated Samples')
            plt.legend()
            plt.show()


def train_score_network_mnist(dataloader, score_net, sde, optimizer, epochs=epochs):
    avg=0
    device = sde.device
    for epoch in range(epochs):

        if(epoch%10==0):
            with torch.no_grad():
                samples = sde.backward_diffusion(score_net, data_shape = (5, 1, 32, 32)).detach().cpu().numpy()
            fig, axe = plt.subplots(5)
            for i in range(5):
                axe[i].imshow(samples[i][0])
            plt.show()

        for x, y in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        print(f'Epoch: {epoch} and Loss: {avg}' )
        avg=0

        torch.save(score_net.state_dict(), f'./epoch{epoch}')


def train_score_network_cifar(dataloader, score_net, sde, optimizer, epochs=epochs):
    avg=0
    device = sde.device
    for epoch in range(epochs):

        if(epoch%10==0):
            with torch.no_grad():
                samples = sde.backward_diffusion(score_net, data_shape = (5, 3, 32, 32)).detach().cpu().numpy()
            samples = samples.swapaxes(1,2)
            samples = samples.swapaxes(2,3)
            fig, axe = plt.subplots(5)
            for i in range(5):
                axe[i].imshow(samples[i])
            plt.show()

        for x, y in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        print(f'Epoch: {epoch} and Loss: {avg}' )
        avg=0

        torch.save(score_net.state_dict(), f'./epoch{epoch}')