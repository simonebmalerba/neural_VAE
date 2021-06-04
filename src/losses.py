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
  p_j_x = encoder(x_data)
  inv_sigma2 = 1/(decoder.sigma.transpose(0,1))**2
  mp = (decoder.mu.transpose(0,1)*inv_sigma2)
  logq_x_j = -0.5*(x_data**2)@inv_sigma2 + (x_data@mp) - 0.5*(mp)*(decoder.mu.transpose(0,1))-         torch.log(np.sqrt(2*np.pi)*decoder.sigma.transpose(0,1))
  loss = -((F.softmax(p_j_x))*logq_x_j).sum(dim=1).mean()
  return loss

def Dkl(x_data,encoder,decoder):
  p_j_x = encoder(x_data)
  dkl = ((F.softmax(p_j_x))*(F.log_softmax(p_j_x) - F.log_softmax(decoder.q))).sum(dim=1).mean() 
  return dkl

def MSE(x_data,encoder,decoder):
  p_j_x = encoder(x_data)
  mse = ((F.softmax(p_j_x))*(x_data**2 + decoder.mu.transpose(0,1)**2 -2*x_data*decoder.mu.transpose(0,1) + decoder.sigma.transpose(0,1)**2)).sum(dim=1).mean()
  return mse
