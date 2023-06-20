#%%
import torch
import torch.nn.functional as F
import numpy as np
import scipy.special
import math
import itertools
from src.encoders_decoders import *
import itertools
import random
# This file contains definitions of loss functions used to train encoder and decoders.
# %%
########################################################################################################
# LOSSES FOR CATEGORICAL ENCODERS/DECODERS (Grechi et al.,2020)
# Simplified losses: encoder is considered ideal given the encoding parameters;
# basically, it corresponds to fit a mixture of Gaussians maximizing the likelihood.

def simplified_loss(x,decoder):
    inv_sigma2 = torch.exp(-2*decoder.log_sigmas.transpose(0,1)) #[1,N]
    mp = decoder.mus.transpose(0,1)*inv_sigma2
    # x has shape [bsize,1]
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.mus.transpose(0,1)-\
    np.log(np.sqrt(2*np.pi)) -decoder.log_sigmas.transpose(0,1) +\
    F.log_softmax(decoder.qs.transpose(0,1),dim=1)
    logZ = -torch.logsumexp(logq_x_j,dim=1)
    return logZ.mean() 

def simplified_loss_orig(x,decoder):
    inv_sigma2 = 1/(decoder.sigma.transpose(0,1))**2
    mp = decoder.mu.transpose(0,1)*inv_sigma2
    q_x_j = -0.5*(x**2)@inv_sigma2 + (x@mp) - 0.5*(mp)*(decoder.mu.transpose(0,1)) -torch.log(np.sqrt(2*np.pi)*decoder.sigma.transpose(0,1)) + torch.log(F.softmax(decoder.q,dim=1))
    logZ = -torch.logsumexp(q_x_j,dim=1)
    return logZ.mean()

# Distortion for a categorical encoder-decoder.
def distortion_cat(x,encoder,decoder):
    # D = E_x[ sum_j p(j|x)*log(q(x|j))]
    p_j_x = F.softmax(encoder(x),dim=1)
    #Compute log(q(x|j))
    inv_sigma2 =torch.exp(-2*(decoder.log_sigmas.transpose(0,1)))
    mp = decoder.mus.transpose(0,1)*inv_sigma2
    logq_x_j = -0.5*(x**2)*inv_sigma2 + x*mp - 0.5*mp*decoder.mus.transpose(0,1)-\
     np.log(np.sqrt(2*np.pi))-decoder.log_sigmas.transpose(0,1)
    D = -((p_j_x*logq_x_j).sum(dim=1)).mean()
    return D

def distortion_cat_orig(x,encoder,decoder):
    p_j_x = encoder(x)
    inv_sigma2 = 1/(decoder.sigma.transpose(0,1))**2
    mp = (decoder.mu.transpose(0,1)*inv_sigma2)
    logq_x_j = -0.5*(x**2)@inv_sigma2 + (x@mp) - 0.5*(mp)*(decoder.mu.transpose(0,1))-torch.log(np.sqrt(2*np.pi)*decoder.sigma.transpose(0,1))
    loss = -((F.softmax(p_j_x))*logq_x_j).sum(dim=1).mean()
    return loss

# Rate for a categorical encoder
def rate_cat(x,encoder,decoder):
    p_tilde = encoder(x)
    R = (F.softmax(p_tilde,dim=1)*(F.log_softmax(p_tilde,dim=1)-\
    F.log_softmax(decoder.qs.transpose(0,1),dim=1))).sum(dim=1).mean()
    return R


def rate_cat_orig(x,encoder,decoder):
    p_j_x = encoder(x)
    dkl = ((F.softmax(p_j_x))*(F.log_softmax(p_j_x) - F.log_softmax(decoder.q))).sum(dim=1).mean() 
    return dkl

# MSE
def MSE_cat(x,decoder,encoder):
    #  Σ_j p(j|x) ((x-μ_j)^2 + σ^2_j)
    if encoder is None:
        encoder = CategoricalEncoder(decoder.mus.transpose(0,1),
        torch.exp(decoder.log_sigmas).transpose(0,1),decoder.qs.transpose(0,1))
    l_j_x = encoder(x)
    mse = (F.softmax(l_j_x,dim=1)*((x-decoder.mus.transpose(0,1))**2 + 
    torch.exp(2*decoder.log_sigmas.transpose(0,1)))).sum(dim=1).mean()
    return mse

def MSE_cat_orig(x,encoder,decoder):
    p_j_x = encoder(x)
    mse = ((F.softmax(p_j_x))*(x**2 + decoder.mu.transpose(0,1)**2 -2*x*decoder.mu.transpose(0,1) + decoder.sigma.transpose(0,1)**2)).sum(dim=1).mean()
    return mse

# %%
########################################################################################################
# LOSSES FOR BERNOULLI ENCODERS/DECODERS.
# Distortion. All functions return D = E_x[E_r[p(r|x)log(q(r|x))]]
def distortion_gaussian(x,encoder,decoder,lat_samp=10,tau=0.5):
    # Gradient of E_r computed with Gumbel-Softmax trick (tau controls steepness of sigmoid function, lat_samp the number of samples
    # of neural activity).
    # Logit r|x.
    l_r_x = encoder(x)
    bsize,N = l_r_x.shape
    #Gumbel Softmax trick.
    eps = torch.rand(bsize,lat_samp,N)
    r = torch.sigmoid((torch.log(eps) - torch.log(1-eps) + l_r_x[:,None,:])/tau)
    mu_dec,log_sigma = decoder(r)
    sigma2_dec = torch.exp(2*log_sigma)
    inv_sigma2_dec = 1/sigma2_dec
    mp = mu_dec*inv_sigma2_dec
    logq_x_r = -0.5*(x**2)*inv_sigma2_dec + x*mp - 0.5*mu_dec*mp -\
        0.5*torch.log(2*np.pi*sigma2_dec)
    D = -logq_x_r.mean()
    return D
 
def distortion_analytical_linear(x,encoder,decoder,r_all):
    # Explicitly compute E_r (limited number of neurons).
    # Requires r_all, the vector of all possible neural responses.
    #Logit r|x
    eta = encoder(x)
    bsize,N = eta.shape
    p_r_x = torch.exp((eta@r_all) - (torch.log( 1 + torch.exp(eta))).sum(dim=1)[:,None])
    mu_dec,log_sigma = decoder(r_all.transpose(0,1)[:,None,:])
    sigma2_dec = torch.exp(2*log_sigma)
    inv_sigma2_dec = 1/sigma2_dec
    mp = mu_dec*inv_sigma2_dec
    logq_x_r = -0.5*(x**2)*inv_sigma2_dec + x*mp - 0.5*mu_dec*mp -\
        0.5*torch.log(2*np.pi*sigma2_dec)
    D = -((p_r_x*logq_x_r).sum(dim=1)).mean()
    return D

# Rate. All functions return R= E_x [KL(p(r|x)/q(r))]]
# Rate as functions.
def rate_iidBernoulli(x,encoder,p_q):
    # Bernoulli prior: q(r) = \prod_i q(r_i).
    # p_q, which can be trainable, are activation priors for each neurons
    l_r_x = encoder(x)
    R = (torch.sigmoid(l_r_x)*(F.logsigmoid(l_r_x) - torch.log(p_q)) + torch.sigmoid(-l_r_x)*(F.logsigmoid(-l_r_x) - torch.log(1-p_q))).sum(dim=1).mean()
    return R

def rate_vampBernoulli(x,encoder,x_k):
    #VAMP prior (Tomczak 2017)
    K,_ = x_k.shape
    l_r_x = encoder(x)[:,:,None]
    l_r = encoder(x_k).transpose(0,1)[None,:,:]
    KLs = (torch.sigmoid(l_r_x)*(F.logsigmoid(l_r_x) - F.logsigmoid(l_r)) + 
    torch.sigmoid(-l_r_x)*(F.logsigmoid(-l_r_x) - F.logsigmoid(-l_r))).sum(dim=1)
    R = -torch.logsumexp(-KLs-np.log(K),dim=1).mean()
    return R

# Rate as classes
class rate_bernoulli(torch.nn.Module):
    # Parameters: bernoulli probabilities (biases)
    def __init__(self,N):
        super().__init__()
        #Biases 1xN
        self.h = torch.nn.Parameter(-1*torch.ones(N)[None,:])
        r_all = np.asarray(list(itertools.product([0, 1], repeat=N)))
        self.r_all = torch.tensor(r_all).transpose(0,1).type(torch.float)
    def forward(self,enc,x):
        eta = enc(x)
        KLs = (torch.sigmoid(eta)*(F.logsigmoid(eta) - F.logsigmoid(self.h)) + 
            torch.sigmoid(-eta)*(F.logsigmoid(-eta) - F.logsigmoid(-self.h))).sum(dim=1)
        R = KLs.mean()
        return R
        
class rate_vampBernoulli(torch.nn.Module):
    def __init__(self,K,x_samples):
        super().__init__()
        self.K = K
        self.x_k = torch.nn.Parameter(x_samples[random.sample(range(
            x_samples.size()[0]),self.K)])
    def forward(self,enc,x):
        l_r_x = enc(x)[:,:,None]
        l_r = enc(self.x_k).transpose(0,1)[None,:,:]
        KLs = (torch.sigmoid(l_r_x)*(F.logsigmoid(l_r_x) - F.logsigmoid(l_r)) + 
        torch.sigmoid(-l_r_x)*(F.logsigmoid(-l_r_x) - F.logsigmoid(-l_r))).sum(dim=1)
        R = -torch.logsumexp(-KLs-np.log(self.K),dim=1).mean()
        return R

class rate_ising(torch.nn.Module):
    # Parameters: biases and couplings of an Ising model
    def __init__(self,N):
        super().__init__()
        #Biases 1xN
        self.h = torch.nn.Parameter(-1*torch.ones(N)[None,:])
        #Initialization of J, NxN matrix. Symmetric and with 0 diagonal.
        #Remember to register the hook for clipping the gradient of diagonal to 0
        W = np.sqrt(1/N)*torch.randn(N,N) 
        J = W*W.transpose(0,1) 
        J.fill_diagonal_(0)
        self.J = torch.nn.Parameter(J)
        #All binary patterns: Nx2^N
        r_all = np.asarray(list(itertools.product([0, 1], repeat=N)))
        self.r_all = torch.tensor(r_all).transpose(0,1).type(torch.float)
    def forward(self,enc,x):
        #Natural parameters of encoder: bsizexN
        eta = enc(x)
        #Mean <r>_r|x
        mu_r_x = torch.sigmoid(eta).transpose(0,1)
        #Data dependent elemts
        eta_h_r = ((eta - self.h)*mu_r_x.transpose(0,1)).sum(dim=1)
        r_J_r = (mu_r_x*(self.J@mu_r_x)).sum(dim=0)
        #Bernoulli partition function   
        logZ1 = (torch.log( 1 + torch.exp(eta))).sum(dim=1)
        #Ising partition function
        logZ = torch.logsumexp((self.h@self.r_all + (self.r_all*(self.J@self.r_all)).sum(dim=0)),1)
        R = (eta_h_r - r_J_r - logZ1 + logZ).mean()
        return R

# %%
def MSE_montecarlo(x,encoder,decoder,lat_samp =10,dec_samp=10):
    r = encoder.sample(x,lat_samp)
    x_dec = decoder.sample(r,dec_samp)
    mseVec = ((x_dec - x[None,:])**2).mean(dim=(0,2))
    return mseVec.mean()
def MSE_posterior_mean(x,encoder,decoder,lat_samp =10,dec_samp=10):
    r = encoder.sample(x,lat_samp)
    x_dec = decoder(r)[0]
    mseVec = ((x_dec - x)**2).mean(dim=1)
    return mseVec.mean()
# %%

