import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC


def match_dim(x, a):
    '''
    Input:
    X -> (B, ...)
    a -> (B)
    Matches a to the shape of b for batch multiplication
    '''
    shape = x.shape
    return a.view(-1, *(1,)*(len(shape)-1))


class SDE(ABC):
    def __init__(self, num_steps, T=1.0, device='cpu'):
        '''
        num_step (number of discretized steps )
        '''
        super().__init__()
        self.num_steps = num_steps
        self.T = T
        self.device = device

    @abstractmethod
    def drift_diffusion(self, x, time):
        '''
        Returns: (drift, diffusion) drift and diffusion coefficient for given samples and timesteps
        '''
        pass

    def marginal(self, x, time):
        '''
        Returns: (mu, std) mariginal distribution of the forward diffusion process x at time t
        '''
        mu, std = self.p(x, time)
        return mu, match_dim(mu, std)

    @abstractmethod
    def p(self, x, time):
        '''
        Returns: (mu, std) mariginal distribution of the forward diffusion process x at time t
        '''
        pass

    @abstractmethod
    def sample_prior(self, shape):
        '''
        Samples the prior distribution for a given shape
        '''
        pass

    def forward_diffusion(self, x, num_steps=None):
        '''
        Numerically performs forward diffusion on the model
        x: input data
        num_steps: number of steps to discretize over (uses self.num_steps as default)
        Returns: x_diffused, time sequence of the diffusion process
        '''
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(
            x, torch.Tensor) else x
        x_diffused = [x.detach().numpy()]  # Store the initial data

        num_steps = self.num_steps if num_steps is None else num_steps

        # Compute indices
        assert num_steps > 0, 'Number of steps must be positive'
        indices = np.arange(num_steps)

        dt = self.T / num_steps  # Calculate the timestep

        time_steps = np.flip((indices + 1) * dt)

        dt = torch.tensor(dt)

        for t in reversed(time_steps):
            # Apply the OU process to the data
            drift, diffusion = self.drift_diffusion(x, t)
            x = x + drift * dt + diffusion * \
                torch.sqrt(dt) * torch.randn_like(x)
            x_diffused.append(x.numpy())
        return x_diffused

    def plot_forward_diffusion(self, data, num_steps=None):
        diffused_data = self.forward_diffusion(data, num_steps)
        num_steps = self.num_steps if num_steps is None else num_steps

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        times = [(i * int(num_steps)) // 7 for i in range(8)]

        for i, ax in enumerate(axes.flatten()):
            z = torch.randn_like(data)
            mu, std = self.marginal(data, torch.tensor(times[i]/num_steps))
            px = mu+std*z
            ax.scatter(px[:, 0], px[:, 1], label=f' P Step {times[i]}')
            ax.scatter(diffused_data[times[i]][:, 0],
                       diffused_data[times[i]][:, 1], label=f'Step {times[i]}')
            ax.set_title(f'Diffusion at step {times[i]}')
            ax.legend()
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def backward_diffusion(self, score_net, data_shape=(1000, 2)):
        '''
        Backward diffusion:
        Score-net: Model to be used
        Data Shape: (Batch, :) specifies how many samples and what shape for the prior distribution
        '''
        device = self.device
        batch_size = data_shape[0]
        x = self.sample_prior(data_shape).to(device)
        dt = torch.tensor(self.T / self.num_steps).to(device)
        indices = torch.arange(self.num_steps).to(device)
        time_steps = torch.flip((indices + 1) * dt, dims=(0,))  # Reverse time

        for t in time_steps:
            t1 = torch.ones(batch_size, device=device) * t
            score = score_net(x, t1)
            drift, diffusion = self.drift_diffusion(x, t)
            x = x - (drift - (diffusion**2)*score)*dt + diffusion * \
                torch.sqrt(dt) * torch.randn_like(x)

        return x


class VPSDE(SDE):

    def __init__(self, num_steps, bmin, bmax, device='cpu'):
        super().__init__(num_steps, T=1.0, device=device)
        self.bmin = bmin
        self.bmax = bmax

    def B(self, t):
        b = self.bmin+t*(self.bmax-self.bmin)
        return b

    def alpha(self, t):
        x = self.bmin*t+((self.bmax-self.bmin)*t**2)/2
        a = torch.exp(-x/2)
        return a

    def p(self, x, t):
        a = self.alpha(t)
        mu = x*match_dim(x, a)
        std = (1-a**2)**0.5
        return mu, std

    def drift_diffusion(self, x, t):
        drift = -self.B(t)/2*x
        diffusion = self.B(t)**.5
        return drift, diffusion

    def sample_prior(self, shape):
        return torch.randn(shape, device=self.device)


'''
def forward_diffusion(data):
    # Convert data to PyTorch tensor if it's not already
    x = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
    x_diffused = [x.numpy()]  # Store the initial data
    

    dt = 1.0 / num_steps  # Calculate the timestep

    indices = np.arange(num_steps)
    time_steps = 1 + indices / (num_steps - 1) * (dt - 1)

    for t in reversed(time_steps):
        # Apply the OU process to the data
        x = x - B(t)/2*x * dt + B(t)**.5 * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x_diffused.append(x.numpy())

    return x_diffused

def plot_forward_diffusion(data):
    diffused_data = forward_diffusion(data)
    print()


    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10)) 
    times = [(i * int(num_steps)) // 7 for i in range(8)]

    for i, ax in enumerate(axes.flatten()):
        z = torch.randn_like(data)
        mu, std=p(data, torch.tensor(times[i]/num_steps).unsqueeze(0)  )
        px=mu+std*z
        ax.scatter(px[:, 0], px[:, 1], label=f' P Step {times[i]}')
        ax.scatter(diffused_data[times[i]][:, 0], diffused_data[times[i]][:, 1], label=f'Step {times[i]}')
        ax.set_title(f'Diffusion at step {times[i]}')
        ax.legend()
        ax.set_aspect('equal') 

    plt.tight_layout() 
    plt.show()

def backward_diffusion(score_net):
    x = torch.randn(num_samples, 2) 
    dt = torch.tensor(1.0 / num_steps)
    indices = torch.arange(num_steps)
    time_steps = 1 + indices / (num_steps - 1) * (dt - 1)

    for t in time_steps:
        t1=torch.ones(num_samples) * t 
        score = score_net(x, t1)  
        x = x + (B(t)/2*x + B(t)*score )*dt + B(t)**.5 * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        
    return x
'''
