import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt



# Function to load configurations
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load the configuration
config = load_config('config.yaml')

num_steps= config['num_steps']
num_samples=config['num_samples']
bmin=config['bmin']
bmax=config['bmax']

def B(t):
    
    b=bmin+t*(bmax-bmin)
    return b

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

def alpha(t):
   x=bmin*t+((bmax-bmin)*t**2)/2
   a=torch.exp(-x/2)
   return a


def p(x, t):
    a=alpha(t)
    a = a.unsqueeze(1)
    mu=x*a
    std=(1-a**2)**0.5
    return mu, std


