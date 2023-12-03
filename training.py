
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


<<<<<<< HEAD
def loss_function(score_net, x, sde, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 
    z = torch.randn_like(x, device = x.device)
    mu, std= sde.marginal(x, random_t)
    perturbed_x=mu+std*z
    score= score_net(perturbed_x, random_t)
=======
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
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a

    '''
    Lambda: retrieves appropriate lambda based off of SDE
    '''
<<<<<<< HEAD
    lamb=1/sde.B(random_t)
    lamb = diffusion.match_dim(x, lamb)
    try:
        loss = torch.mean(lamb*torch.square((std*score  + z)))
    except:
        print(lamb.shape, std.shape, score.shape, z.shape)
        ## Error catching

    return loss

def train_score_network(dataloader, score_net, sde, epochs=epochs):
    optimizer = optim.Adam(score_net.parameters(), lr=learning_rate)
    avg=0
    for epoch in range(epochs):
=======
    lamb = 1/sde.B(random_t)
    lamb = lamb.unsqueeze(1)

    # print(lamb.shape, std.shape, score.shape, z.shape)
    loss = torch.mean(lamb*torch.square((std*score + z)))

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
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a
        for x_batch, in dataloader:
            optimizer.zero_grad()
            loss = loss_function(score_net, x_batch, sde)
            loss.backward()
            # nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
<<<<<<< HEAD
            avg+=loss
        if (epoch%100==0):
            print(f'Epoch: {epoch} and Loss: {avg/(8*100)}' )
            avg=0
        if(epoch%1000==0 and epoch!=0):
            samples = sde.backward_diffusion(score_net)
            data=x_batch.detach().numpy()
=======
            avg += loss
        if (epoch % 100 == 0):
            tqdm.write(f'Epoch: {epoch} and Loss: {avg/(8*100)}')
            avg = 0
        if ((epoch % 1000 == 0 and epoch != 0) or epoch == epochs-1):
            samples = sde.backward_diffusion(score_net)
            data = x_batch.detach().numpy()
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a
            samples_np = samples.detach().numpy()
            plt.scatter(data[:, 0], data[:, 1], label='Original Data')
            plt.scatter(samples_np[:, 0], samples_np[:,
                        1], label='Generated Samples')
            plt.legend()
            plt.show()

<<<<<<< HEAD
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
=======

def train_score_network_mnist(dataloader, score_net, sde, epochs=epochs):
    optimizer = get_optimizer(score_net)
    device = sde.device
    for epoch in tqdm(range(epochs)):
        avg = 0
        for x, _ in dataloader:
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
<<<<<<< HEAD
            avg+=loss
        print(f'Epoch: {epoch} and Loss: {avg}' )
        avg=0
=======
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
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a

        torch.save(score_net.state_dict(), f'./epoch{epoch}')


<<<<<<< HEAD
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
=======
def train_score_network_cifar(dataloader, score_net, sde, epochs=epochs):
    optimizer = get_optimizer(score_net)
    device = sde.device
    for epoch in tqdm(range(epochs)):
        avg = 0
        for x, _ in dataloader:
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_function(score_net, x, sde)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
<<<<<<< HEAD
            avg+=loss
        print(f'Epoch: {epoch} and Loss: {avg}' )
        avg=0

        torch.save(score_net.state_dict(), f'./epoch{epoch}')
=======
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
>>>>>>> c46bb0fc2c1ea9938b25531f41be428ffca1813a
