#%%
import torch
import torch.nn.functional as F
import numpy as np
# %%
## Encoders
# Proposed convention: x is a vector [bsize_size,1] (or  [bsize_size, x_dim]).
# Encoder should return a [bsize_size,N] tensor as a differentiable function of the encoding parameters (e.g. logits, probs).
# Sample method is used to sample from the latent space, but is not differentiable.
def initialize_categorical_params(c0,sigma0,q0):
    a = torch.nn.Parameter(-1/(2*sigma0**2))
    b = torch.nn.Parameter(c0/sigma0**2)
    c = torch.nn.Parameter(-c0**2/(2*sigma0**2) -torch.log(sigma0) - torch.log(F.softmax(q0,dim=1)))
    return a,b,c
def initialize_bernoulli_params(N,x_min,x_max):
    #Initialize parameters arranging centers equally spaced in the range x_min x_max,
    # and the width as 5 times the spacing between centers
    cs = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-x_min)/N))[None,:]
    log_sigmas = torch.log(torch.ones(N)*5*(x_max-x_min)/N)[None,:]
    As = torch.nn.Parameter(torch.ones(N))[None,:]
    return cs,log_sigmas,As
class CategoricalEncoder(torch.nn.Module):
    #It returns for each stimulus x a vector of (unnormalized) probabilities (i.e.logits) of activation for
    #each neuron, which is a quadratic function of x. 
    def __init__(self,c0,sigma0,q0):
        super().__init__()
        self.a,self.b,self.c = initialize_categorical_params(c0,sigma0,q0)
    def forward(self,x):
        # x has shape [bsize_size, xdim], a,b,c has shape [xdim, N]
        p_tilde = (x**2)@(self.a) + x@(self.b) + self.c
        return p_tilde
    def sample(self,x,nsamples):
        p_r_x = F.softmax(self.forward(x),dim=1)
        return torch.distributions.categorical.Categorical(p_r_x).sample((nsamples,))


class BernoulliEncoder(torch.nn.Module):
    # Encoder returning for N neurons their unnormalized probabilities of being active (i.e. logits),as 
    # a quadratic function of x
    def __init__(self,N,x_min,x_max):
        super().__init__()
        self.cs, self.log_sigmas,self.As  = initialize_bernoulli_params(N,x_min,x_max)
    def forward(self,x):
        # x has shape [bsize_dim,x_dim], c,log_sigma,A have shape [x_dim, N]
        inv_sigmas = 0.5*torch.exp(-2*self.log_sigmas)
        etas = -(x**2)@inv_sigmas + 2*x@(self.cs*inv_sigmas) - (self.cs**2)*inv_sigmas + torch.log(self.As)
        return etas
    def sample(self,x,nsamples):
        r = torch.distributions.bernoulli.Bernoulli(logits = self.forward(x)).sample((nsamples,))
        return r

# %%
## Decoders
# After a sampling from the latent space, we have to get a matrix [bsize,n_samples,N] of neural activation.
# The decoder may have different forms, but should return the parameters of an output distribution.
def initialize_MOG_params(N,x_min,x_max):
    #Initialize parameters arranging centers equally spaced in the range x_min x_max,
    # and the width as 5 times the spacing between centers
    mus = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-x_min)/N))[:,None]
    log_sigmas = torch.log(torch.ones(N)*5*(x_max-x_min)/N)[:,None]
    qs = torch.nn.Parameter(torch.ones(N))[:,None]
    return qs,mus,log_sigmas
class MoGDecoder(torch.nn.Module):
    #Decoder as a mixture of Gaussians, parameters are q(memberships), mean and variances.
    def __init__(self,N,x_min,x_max):
        super().__init__()
        self.qs,self.mus,self.log_sigmas = initialize_MOG_params(N,x_min,x_max)
    def forward(self,r):
        # r is a tensor [bsize,lat_samples,N],mus and sigmas are [N,1]. Note that input 
        # should be a one_hot tensor. Alternative is 
        # return self.mus[r],self.log_sigmas[r] if r are categories
        return r @ self.mus, r @ self.log_sigmas

class GaussianDecoder(torch.nn.Module):
    # Return natural parameters of a gaussian distribution as a linear combination of neural activities
    def __init__(self,phi0):
        super().__init__()
        self.phi = torch.nn.Parameter(phi0)
        self.b = torch.nn.Parameter(torch.tensor([0,-1e-7]))
        #self.A = torch.nn.Parameter(A0)
    def forward(self,r):
        # r must have shape [bsize,lat_samples,N], phi have shape [N,2].
        # mu and sigma are [bsize,lat_samples]
        eta = r @ self.phi + self.b
        eta1,eta2 = eta[:,:,0],eta[:,:,1]
        mu = -0.5*eta1/eta2
        sigma =  - 0.5/eta2
        return mu, sigma


# %%
