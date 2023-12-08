import torch
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod, ABC
from guided_diffusion import cross_entropy_loss_function


def match_dim(x, a):
    '''
    Match Dimensions for Batch Multiplication. 

    Inputs:
    - x: Tensor of shape (B, ...) where B is the batch size
    - a: Tensor of shape (B)

    Returns:
    - Tensor: Reshaped version of 'a' (by adding singleton dimensions) to match the shape of 'x' for batch multiplication.

    Example:
    If x.shape = (B, C, H, W) and a.shape = (B), this function will reshape 'a' to have dimensions (B, 1, 1, 1) to match 'x' for batch multiplication.
    '''
    shape = x.shape
    return a.view(-1, *(1,)*(len(shape)-1))


class SDE(ABC):
    def __init__(self, num_steps, T, device='cpu'):
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

    def get_dt_time_steps(self, device=None):
        '''
        Returns the dt and time steps for the diffusion process
        '''
        dt = torch.tensor(self.T / self.num_steps)
        indices = torch.arange(self.num_steps)
        if device is not None:
            dt = dt.to(device)
            indices = indices.to(device)
        time_steps = torch.flip((indices + 1) * dt, dims=(0,))  # Reverse time
        return dt, time_steps

    def forward_diffusion(self, x, num_steps=None):
        '''
        Numerically performs forward diffusion on the model
        x: input data
        num_steps: number of steps to discretize over (uses self.num_steps as default)
        Returns: x_diffused, time sequence of the diffusion process
        '''
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        x_diffused = [x.detach().numpy()]  # Store the initial data

        num_steps = self.num_steps if num_steps is None else num_steps

        dt, time_steps = self.get_dt_time_steps()

        for t in reversed(time_steps):
            # Apply the OU process to the data
            drift, diffusion = self.drift_diffusion(x, t)
            x = x + drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
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
            ax.scatter(diffused_data[times[i]][:, 0], diffused_data[times[i]][:, 1], label=f'Step {times[i]}')
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
        dt, time_steps = self.get_dt_time_steps(device=device)

        for t in time_steps:
            t1 = torch.ones(batch_size, device=device) * t
            score = score_net(x, t1)
            drift, diffusion = self.drift_diffusion(x, t)
            x = x - (drift - (diffusion**2)*score)*dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)

        return x

    def classifier_guided_backward_diffusion(self, score_net, classifier_net, data_shape=(1000, 2), classes=None):
        '''
        Backward diffusion:
        Score-net: Model to be used
        Data Shape: (Batch, :) specifies how many samples and what shape for the prior distribution
        '''

        assert (classes.shape[0] == data_shape[0])

        device = self.device
        batch_size = data_shape[0]
        x = self.sample_prior(data_shape).to(device)
        dt, time_steps = self.get_dt_time_steps(device=device)
        x.requires_grad = True
        for t in time_steps:
            t1 = torch.ones(batch_size, device=device) * t
            score = score_net(x, t1)
            drift, diffusion = self.drift_diffusion(x, t)

            # Calculate the gradient with respect to x

            pred = classifier_net(x, t1)
            # print('predictions:', torch.argmax(pred, dim = 1))
            pred = torch.log(pred)
            # print(pred.shape, torch.nn.functional.one_hot(classes, num_classes = 10).shape)
            pred = (torch.nn.functional.one_hot(
                classes, num_classes=10) * pred).sum(-1)
            x_grad = [torch.autograd.grad(outputs=out, inputs=x, retain_graph=True)[
                0][i] for i, out in enumerate(pred)]

            x = x - (drift - (diffusion**2)*(score+torch.stack(x_grad)))*dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)

        return x

    def infilling_diffusion(self, score_net, partial_data, mask, data_shape=(1000, 2)):
        '''
        Backward diffusion:
        Score-net: Model to be used
        Data Shape: (Batch, :) specifies how many samples and what shape for the prior distribution

        X: the original data
        Mask: the mask specifying what is known in the data
        '''
        device = self.device
        batch_size = data_shape[0]
        x = self.sample_prior(data_shape).to(device)
        dt, time_steps = self.get_dt_time_steps(device=device)

        for t in time_steps:
            t1 = torch.ones(batch_size, device=device) * t
            score = score_net(x, t1)
            drift, diffusion = self.drift_diffusion(x, t)
            x = x - (drift - (diffusion**2)*score)*dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)

            # substitute the infilling portion with a valid x
            mu, std = self.marginal(partial_data, t1)
            z = torch.randn_like(partial_data, device=x.device)
            partial = mu + std*z
            x = x*(1-mask) + mask*partial_data

        return x


class VPSDE(SDE):

    def __init__(self,  num_steps, bmin, bmax, logarithmic_scheduling=False, device='cpu'):
        super().__init__(num_steps,  T=1.0, device=device)
        self.bmin = bmin
        self.bmax = bmax
        self.logarithmic_scheduling = logarithmic_scheduling

    def B(self, t):
        if self.logarithmic_scheduling:
            r = torch.log(torch.tensor(self.bmax / self.bmin)) / self.T
            return self.bmin * torch.exp(r * t)
        else:
            return self.bmin+t*(self.bmax-self.bmin)

    def alpha(self, t):
        x = None
        if self.logarithmic_scheduling:
            r = torch.log(torch.tensor(self.bmax / self.bmin)) / self.T
            x = (self.bmin / r) * (torch.exp(r * t) - 1)
        else:
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


class BridgeDiffusionVPSDE(SDE):
    def __init__(self, data_y, num_steps=1000, num_samples=1000, bmin=.1, bmax=.1, device='cpu'):
        super().__init__(num_steps, T=1.0, device=device)
        # Initialize additional parameters for Bridge Diffusion
        self.bmin = bmin
        self.bmax = bmax
        self.data_y = data_y.to(device)
        self.num_samples = num_samples

    def B(self, t):
        b = self.bmin+t*(self.bmax-self.bmin)
        return b.to(self.device)

    def alpha(self, t):
        x = self.bmin*t+((self.bmax-self.bmin)*t**2)/2
        a = (torch.exp(-x/2)).to(self.device)
        return a

    def sigma(self, t):
        std = ((1-self.alpha(t)**2)**0.5).to(self.device)
        return std

    def SNR(self, t):
    
        return (self.alpha(t)**2/self.sigma(t)**2).to(self.device)

    def p(self, x, t, y, T=torch.tensor(1)):
        y=y.to(self.device)
        x=x.to(self.device)
        t = t.unsqueeze(-1)
        t=match_dim(x, t).to(self.device)
        T=match_dim(x, T).to(self.device)
        mu = y*(self.SNR(T)/self.SNR(t))*(self.alpha(t)/self.alpha(T)) + self.alpha(t)*x*(1-self.SNR(T)/self.SNR(t))
        std = self.sigma(t)*torch.sqrt(1.-(self.SNR(T)/self.SNR(t)))
        return mu.to(self.device), std.to(self.device)

    def h(self, x, t, y, T=torch.tensor(1)):
        t=match_dim(x, t).to(self.device)
        T=match_dim(x, T).to(self.device)
        # Correction term for bridge diffusion
        score = ((self.alpha(t)/self.alpha(T))*y-x) / (self.sigma(t)**2*(self.SNR(t)/self.SNR(T)-1))
        return score

    def g(self, t):
        # diffusion
        g = self.B(t)**.5
        return g.to(self.device)
    def f(self, x, t):
        # drift
        f = x*-self.B(t)/2
        return f.to(self.device)

    def drift_diffusion(self, x, t):
        drift = self.f(x, t)
        diffusion = self.g(t)
        return drift.to(self.device), diffusion.to(self.device)

    def sample_prior(self, shape):
        return self.data_y(shape).to(self.device)

    def marginal(self, x, time, y):
        '''
        Returns: (mu, std) mariginal distribution of the forward diffusion process x at time t
        '''
        mu, std = self.p(x, time, y)
        return mu, match_dim(mu, std)

    def forward_diffusion(self, data_x, data_y):
        # Convert data to PyTorch tensor if it's not already
        x = torch.tensor(data_x, dtype=torch.float32) if not isinstance(data_x, torch.Tensor) else data_x
        y = torch.tensor(data_y, dtype=torch.float32) if not isinstance(data_y, torch.Tensor) else data_y
        x_diffused = [x.numpy()]  # Store the initial data
        dt, time_steps = self.get_dt_time_steps()
        # ensure that the time_steps are larger than 0
        assert torch.all(time_steps > 0), "Time steps should be larger than 0"
        for t in time_steps:
            t=torch.tensor(t).to(self.device)
            x = x + (self.f(x, t)+self.g(t)**2*self.h(x, t, y))*dt + self.g(t) * torch.sqrt(dt) * torch.randn_like(x)
            x_diffused.append(x.numpy())

        return x_diffused

    def plot_forward_diffusion(self, data_x):
        data_y = self.sample_prior(data_x.shape[0])
        diffused_data = self.forward_diffusion(data_x, data_y)
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        times = [(i * int(self.num_steps)) // 7 for i in range(8)]
        for i, ax in enumerate(axes.flatten()):
            z = torch.randn_like(data_x)
            mu, std = self.p(data_x, torch.tensor(times[i]/self.num_steps), data_y)
            px = mu+std*z
            ax.scatter(px[:, 0], px[:, 1], label=f' P Step {times[i]}')
            ax.scatter(diffused_data[times[i]][:, 0], diffused_data[times[i]][:, 1], label=f'Step {times[i]}')
            ax.set_title(f'Diffusion at step {times[i]}')
            ax.legend()
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def backward_diffusion(self, score_net, data_shape=(1000, 2)):
        device = self.device
        batch_size = data_shape[0]
        y = self.sample_prior(batch_size)
        x = y
        dt, time_steps = self.get_dt_time_steps(device=device)
        for t in time_steps:
            x_and_y = torch.cat((x, y), dim=1)
            t1 = torch.ones(batch_size, device=device) * t
            score = score_net(x_and_y, t1)
            drift, diffusion = self.drift_diffusion(x, t)
            x = x - (drift - (diffusion**2)*((score)-self.h(x, t, y)))*dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
        return x
