#%%
import torch
import torch.nn.functional as F
import numpy as np
from src.encoders_decoders import *
# %%
### Categorical latent space 

def simplified_loss(x,decoder):
    # Compute loss function of MoG decoder in the ideal case of optimal encoder
    # given the decoder , as log(Σ_j q_j q(x|j))
    inv_sigma2 = torch.exp(-2*decoder.log_sigmas.transpose(0,1)) #[1,N]
    mp = decoder.mus.transpose(0,1)*inv_sigma2
    # x has shape [bsize,1]d
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.mus.transpose(0,1)-\
    np.log(np.sqrt(2*np.pi)) -decoder.log_sigmas.transpose(0,1) +\
    F.log_softmax(decoder.qs.transpose(0,1),dim=1)
    logZ = -torch.logsumexp(logq_x_j,dim=1)
    return logZ.mean() 

#Non ideal encoder 
def distortion_cat(x,encoder,decoder):
    # E_x[ sum_j p(j|x)*log(q(x|j))]
    p_j_x = F.softmax(encoder(x),dim=1)
    #Compute log(q(x|j))
    inv_sigma2 =torch.exp(-2*(decoder.log_sigmas.transpose(0,1)))
    mp = decoder.mus.transpose(0,1)*inv_sigma2
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.mus.transpose(0,1)-\
    np.log(np.sqrt(2*np.pi))-decoder.log_sigmas.transpose(0,1)
    D = -((p_j_x*logq_x_j).sum(dim=1)).mean()
    return D

def rate_cat(x,encoder,decoder):
    p_tilde = encoder(x)
    R = (F.softmax(p_tilde,dim=1)*(F.log_softmax(p_tilde,dim=1)-\
    F.log_softmax(decoder.qs.transpose(0,1),dim=1))).sum(dim=1).mean()
    return R
def MSE_cat(x,decoder,encoder=None):
    if encoder is None:
        encoder = CategoricalEncoder(decoder.mus.transpose(0,1),torch.exp(decoder.log_sigmas).transpose(0,1),decoder.qs.transpose(0,1))
    l_j_x = encoder(x)
    mse = (F.softmax(l_j_x,dim=1)*((x-decoder.mus.transpose(0,1))**2 + torch.exp(2*decoder.log_sigmas.transpose(0,1)))).sum(dim=1).mean()
    return mse
# %%
#Bernoulli latent space
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


def rate(x):
    a = x**2
    return a
    
