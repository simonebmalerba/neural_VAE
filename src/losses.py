#%%
import torch
import torch.nn.functional as F
from encoders_decoders import *
# %%
N = 20
c0 = torch.arange(0,10,10/N)[None,:]
sigma0 = 0.5*torch.ones(N)[None,:]
#q0 = 1/N*torch.ones(N)[None,:]
A0 = torch.ones(N)[None,:]
#Define data distribution
x_data = torch.distributions.exponential.Exponential(0.5).sample((500,))[:,None]
x_sorted,indices = x_data.sort(dim=0)
x_min,x_max = x_sorted[0,:].item(),x_sorted[-1,:].item() 
# %%
def distortion(x,encoder,decoder):