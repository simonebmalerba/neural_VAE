#%%
import torch
import torch.nn.functional as F
import numpy as np
from encoders_decoders import *
# %%
### Categorical latent space 

def simplified_loss(x,decoder):
    # Compute loss function of MoG decoder in the ideal case of optimal encoder
    # given the decoder , as log(Σ_j q_j q(x|j))
    inv_sigma2 = torch.exp(-2*decoder.log_sigmass.transpose(0,1)) #[1,N]
    mp = decoder.muss.transpose(0,1)*inv_sigma2
    # x has shape [bsize,1]d
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.muss.transpose(0,1)
    -np.log(np.sqrt(2*np.pi)) -decoder.log_sigmas.transpose(0,1)
    + torch.LogSoftmax(decoder.q,dim=1)
    logZ = -torch.logsumexp(logq_x_j,dim=1)
    return logZ.mean() 

#Non ideal encoder 
def distortion_cat(x,encoder,decoder):
    # E_x[ sum_j p(j|x)*log(q(x|j))]
    p_j_x = F.softmax(encoder(x),dim=1)
    #Compute log(q(x|j))
    inv_sigma2 =torch.exp(-2*(decoder.log_sigmass.transpose(0,1)))
    mp = (decoder.mus.transpose(0,1)*inv_sigma2)
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.mus.transpose(0,1)
    - np.log(np.sqrt(2*np.pi))-decoder.log_sigmas.transpose(0,1)
    D = -((p_j_x*logq_x_j).sum(dim=1)).mean()
    return D

def rate_cat(x,encoder,decoder):
    p_tilde = encoder(x)
    R = (F.softmax(p_tilde)*(F.log_softmax(p_tilde) 
    - F.log_softmax(decoder.q))).sum(dim=1).mean()
    return R
#
# %%
def distortion_gaussian(x,encoder,decoder,lat_samp=10,tau=0.5):
    p_r_x = encoder(x)
    bsize,N = p_r_x.shape
    eps = torch.rand(bsize,lat_samp,N)
    r = torch.sigmoid((torch.log(eps) - torch.log(1-eps) + p_r_x[:,None,:])/tau)
    mu_dec,sigma2_dec = decoder(r)
    inv_sigma2_dec = 1/sigma2_dec
    mp = mu_dec*inv_sigma2_dec
    logq_x_r = -0.5*(x**2)*inv_sigma2_dec + x*mp - 0.5*mu_dec*mp - 0.5*torch.log(2*np.pi*sigma2_dec)
    D = -logq_x_r.mean()
    return D
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
