#%%
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import cluster
# THis file contains definitions of encoding and decoding functions.
# %% 
# #############################################################################
# ENCODER DEFINITIONS
# Convention: x is a vector [bsize_size,1] (or  [bsize_size, x_dim]).
# Encoder returns a [bsize_size,N] tensor of neural responses/latent variables
# as a differentiable function of encoding parameters (e.g. logits, probs). 
# Only forward method is used during training; sample method is used to sample  from the latent space, but is not differentiable. 

# Initialization of tuning curve parameters (centers, tuning width and amplitudes) of the encoder.
def initialize_categorical_params(c0,sigma0,q0):
    a = torch.nn.Parameter(-1/(2*sigma0**2))
    b = torch.nn.Parameter(c0/sigma0**2)
    c = torch.nn.Parameter(-c0**2/(2*sigma0**2) -torch.log(sigma0) - 
    torch.log(F.softmax(q0,dim=1)))
    return a,b,c

def initialize_bernoulli_params(N,x_min,x_max,xs,w=1,z=0.01):
    # Initalize preferred positions with kmean algorithm on the data points.
    # initalize sigmas to be proprotional (factor of w) to spacing between centers
    kmeans = cluster.KMeans(n_clusters=N, init='random',
                        n_init=10, max_iter=10, random_state=2)
    C = kmeans.fit_predict(xs)
    centers = kmeans.cluster_centers_
    cs = torch.nn.Parameter(torch.Tensor(centers).transpose(0,1))
    cs_sorted,indices = cs.sort(dim=1)
    indices = torch.squeeze(indices)
    idr = np.argsort(indices)
    deltac = torch.diff(cs_sorted)
    deltac = torch.cat((deltac[0,:],deltac[:,-1]))
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*torch.sqrt(w*deltac))[None,idr])
    As = torch.nn.Parameter(torch.ones(N)[None,:])
    return cs,log_sigmas,As

class CategoricalEncoder(torch.nn.Module):
    # Categorical encoder: for each stimulus, x, returns the probabilities (actually, logits)
    # of a N-dimensional categorical distribution (probability of activation of each neuron),
    # as paramterized quadratic functions of the stimulus (with parameters centers, tuning widths and amplitude of tuning curves).
    # A population response  thus consists in a vector with a single neuron active (as in Grechi et al.,2020).
    def __init__(self,c0,sigma0,q0):
        super().__init__()
        self.a,self.b,self.c = initialize_categorical_params(c0,sigma0,q0)
    def forward(self,x):
        # x has shape [bsize_size, xdim], a,b,c has shape [xdim, N]
        p_tilde = (x**2)@(self.a) + x@(self.b) + self.c
        return p_tilde
    def sample(self,x,nsamples):
        _,N = self.a.shape
        p_r_x = F.softmax(self.forward(x),dim=1)
        r_cat = torch.distributions.categorical.Categorical(p_r_x).sample((nsamples,))
        r = F.one_hot(r_cat,N).to(dtype=torch.float32)
        return r

class BernoulliEncoder(torch.nn.Module):
    # Bernoulli encoder: for each stimulus, x, returns the probabilities (actually, logits)
    # of N Bernoulli distributions as parametrized quadratic functions of the stimulus
    # (with parameters centers, tuning widths and amplitude of tuning curves).
    # A population response consists in a vector of activation of neurons sampled independently from the Bernoulli
    # distributions.
    def __init__(self,N,x_min,x_max,xs,w=1):
        super().__init__()
        self.cs, self.log_sigmas,self.As  = initialize_bernoulli_params(N,x_min,x_max,xs,w=w)
    def forward(self,x):
        # x has shape [bsize_dim,x_dim], c,log_sigma,A has shape [x_dim, N]
        inv_sigmas = 0.5*torch.exp(-2*self.log_sigmas)
        etas1 = -(x**2)@inv_sigmas
        etas2 = + 2*x@((self.cs*inv_sigmas))
        etas3 = - (self.cs**2)*inv_sigmas + torch.log(self.As)
        return etas1 + etas2 + etas3

    def sample(self,x,nsamples):
        p_r_x = torch.distributions.bernoulli.Bernoulli(logits = self.forward(x))
        r = p_r_x.sample((nsamples,)).transpose(0,1)
        return r

# %%
# ###############################################################################
# DECODER DEFINITIONS
# A decoder takes in input a sample of neural responses (a matrix [bsize,n_samples,N]) and outputs parameters 
# of decoding distribution in the stimulus space (e.g., mean and variance of a Gaussian distribution). 
# Only forward method is used during training procedure; sample method can be used to sample stimulus estimates.

#Parameters initializations
def initialize_MOG_params(N,x_min,x_max, x):
    #Initialize parameters arranging centers equally spaced in the range x_min x_max,
    # and the width as 5 times the spacing between centers.
    kmeans = cluster.KMeans(n_clusters=N, init='random',
                        n_init=10, max_iter=10, random_state=2)
    C = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_
    mus = torch.nn.Parameter(torch.Tensor(centers))

    #mus = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-x_min)/N)[:,None])
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*(x_max-x_min)/N)[:,None])
    qs = torch.nn.Parameter(torch.ones(N)[:,None])
    return qs,mus,log_sigmas

class MoGDecoder(torch.nn.Module):
    # Decoder as a mixture of Gaussians, parameters are q(memberships), mean and variances.
    # This decoder is paired to the categorical encoder, and performs a mixture of gaussians fit (see Grechi et al.,2020).
    def __init__(self,N,x_min,x_max,x):
        super().__init__()
        self.qs,self.mus,self.log_sigmas = initialize_MOG_params(N,x_min,x_max,x)
    def forward(self,r):
        # r is a tensor [bsize,lat_samples,N],mus and sigmas are [N,1]. Note that input 
        # should be a one_hot tensor. 
        return r @ self.mus, r @ self.log_sigmas
    def sample(self,r,dec_samples):
        mu_dec,log_sigma_dec = self.forward(r)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.exp(log_sigma_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec    

class MLPDecoder(torch.nn.Module):
    # Multilayer perceptron with one hidden layer, which returns mean and (log) variance of 
    # a Gaussian distribution.
    def __init__(self,N,M):
        super().__init__()
        self.hidden = torch.nn.Linear(N,M)
        self.f = torch.nn.ReLU()
        self.w = torch.nn.Linear(M,2)
    def forward(self,r):
        H = self.f(self.hidden(r))
        mu,log_sigma = torch.split(self.w(H),1,dim=2)
        return torch.squeeze(mu),torch.squeeze(log_sigma)
    def sample(self,r,dec_samples):
        mu_dec,log_sigma = self.forward(r)
        sigma2_dec = torch.exp(2*log_sigma)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.sqrt(sigma2_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec
    
class MLPDecoder2n(torch.nn.Module):
    # Multilayer Perceptron with two different hidden layer for mu and sigma.
    def __init__(self,N,M):
        super().__init__()
        self.hidden = torch.nn.Linear(N,M)
        self.hidden2 = torch.nn.Linear(N,M)
        self.f = torch.nn.ReLU()
        self.wm = torch.nn.Linear(M,1)
        self.ws = torch.nn.Linear(M,1)
    def forward(self,r):
        H = self.f(self.hidden(r))
        H2 = self.f(self.hidden2(r))
        mu = self.wm(H)
        log_sigma  = self.ws(H2)
        return torch.squeeze(mu),torch.squeeze(log_sigma)
    def sample(self,r,dec_samples):
        mu_dec,log_sigma = self.forward(r)
        sigma2_dec = torch.exp(2*log_sigma)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.sqrt(sigma2_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec