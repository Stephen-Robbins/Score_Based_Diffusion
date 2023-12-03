
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from prodigyopt import Prodigy


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config('config.yaml')
num_steps = config['num_steps']
num_samples = config['num_samples']
# bmin=config['bmin']
# bmax=config['bmax']
learning_rate = config['learning_rate']
epochs = config['epochs']
optimizer = config['optimizer']


def loss_function(score_net, x, sde, eps=1e-5):
    """"
    Loss function for score matching
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x, device=x.device)
    mu, std = sde.marginal(x, random_t)
    perturbed_x = mu+std*z
    score = score_net(perturbed_x, random_t)

    '''
    Lambda: retrieves appropriate lambda based off of SDE
    '''
    lamb=1/sde.B(random_t)
    lamb = lamb.unsqueeze(1)

    loss = torch.mean(lamb*torch.square((std*score  + z)))
    
    return loss


def get_optimizer(model):
    """
    Returns the optimizer based on the config file
    """
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        # return Prodigy(model.parameters())
        return Prodigy(model.parameters(), safeguard_warmup=True, use_bias_correction=True, weight_decay=0.01)


def train_score_network(dataloader, score_net, sde, epochs=epochs):
    """
    Trains the score network

    """

    optimizer = get_optimizer(score_net)
    avg = 0
    for epoch in tqdm(range(epochs)):
        for x_batch, in dataloader:
            optimizer.zero_grad()
            loss = loss_function(score_net, x_batch, sde)
            loss = loss_function(score_net, x_batch, sde)
            loss.backward()
            # nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            # nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg+=loss
        if (epoch%100==0):
            print(f'Epoch: {epoch} and Loss: {avg/(8*100)}' )
            avg=0
        if(epoch%1000==0 and epoch!=0):
            samples = sde.backward_diffusion(score_net)
            data=x_batch.detach().numpy()
            avg += loss
        if (epoch % 100 == 0):
            tqdm.write(f'Epoch: {epoch} and Loss: {avg/(8*100)}')
            avg = 0
        if ((epoch % 1000 == 0 and epoch != 0) or epoch == epochs-1):
            samples = sde.backward_diffusion(score_net)
            data = x_batch.detach().numpy()
            samples_np = samples.detach().numpy()
            plt.scatter(data[:, 0], data[:, 1], label='Original Data')
            plt.scatter(samples_np[:, 0], samples_np[:,
                        1], label='Generated Samples')
            plt.scatter(samples_np[:, 0], samples_np[:,
                        1], label='Generated Samples')
            plt.legend()
            plt.show()


def train_score_network_mnist(dataloader, score_net, sde, epochs=epochs):
    optimizer = get_optimizer(score_net)
    device = sde.device
    for epoch in tqdm(range(epochs)):
        avg = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg += loss
        print(f'Epoch: {epoch} and Loss: {avg}')

        n_examples = 5  # number of examples to generate
        with torch.no_grad():
            samples = sde.backward_diffusion(
                score_net, data_shape=(n_examples, 1, 32, 32)).detach().cpu().numpy()

        _, axes = plt.subplots(1, n_examples)

        for i, ax in enumerate(axes):
            ax.imshow(samples[i].squeeze())
            ax.axis('off')

        plt.show()

        torch.save(score_net.state_dict(), f'./epoch{epoch}')


def train_score_network_cifar(dataloader, score_net, sde, epochs=epochs):
    optimizer = get_optimizer(score_net)
    device = sde.device
    for epoch in tqdm(range(epochs)):
        avg = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            avg += loss
        print(f'Epoch: {epoch} and Loss: {avg}')

        n_examples = 5  # number of examples to generate
        with torch.no_grad():
            samples = sde.backward_diffusion(
                score_net, data_shape=(n_examples, 3, 32, 32)).detach().cpu().numpy()
        samples = samples.swapaxes(1, 2)
        samples = samples.swapaxes(2, 3)
        _, axes = plt.subplots(1, n_examples)

        for i, ax in enumerate(axes):
            ax.imshow(samples[i].squeeze())
            ax.axis('off')

        plt.show()

        torch.save(score_net.state_dict(), f'./epoch{epoch}')
