#%%
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import cluster
# %%
## Encoders
# Proposed convention: x is a vector [bsize_size,1] (or  [bsize_size, x_dim]).
# Encoder should return a [bsize_size,N] tensor as a differentiable function of 
# the encoding parameters (e.g. logits, probs). Sample method is used to sample 
# from the latent space, but is not differentiable. 
def initialize_categorical_params(c0,sigma0,q0):
    a = torch.nn.Parameter(-1/(2*sigma0**2))
    b = torch.nn.Parameter(c0/sigma0**2)
    c = torch.nn.Parameter(-c0**2/(2*sigma0**2) -torch.log(sigma0) - 
    torch.log(F.softmax(q0,dim=1)))
    return a,b,c

def initialize_bernoulli_params(N,x_min,x_max,xs):
    #Initialize parameters arranging centers equally spaced in the range x_min x_max,
    # and the width as 5 times the spacing between centers
    #initialize cs using kmeans:
    kmeans = cluster.KMeans(n_clusters=N, init='random',
                        n_init=10, max_iter=10, random_state=2)
    C = kmeans.fit_predict(xs)
    centers = kmeans.cluster_centers_
    cs = torch.nn.Parameter(torch.Tensor(centers).transpose(0,1))
    #cs = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-x_min)/N)[None,0:N])
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*(x_max-x_min)/N)[None,:])
    As = torch.nn.Parameter(torch.ones(N)[None,:])
    return cs,log_sigmas,As

def initialize_bernoulli_params_sigma(N,x_min,x_max,xs,w):
    # w = new parameter used in the definition of the curve's width 
    #cs are initialized using kmeans:
    kmeans = cluster.KMeans(n_clusters=N, init='random', n_init=10, max_iter=10, random_state=2)
    C = kmeans.fit_predict(xs)
    centers = kmeans.cluster_centers_
    cs = torch.nn.Parameter(torch.Tensor(centers).transpose(0,1))
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*(x_max-x_min)/w)[None,:])
    As = torch.nn.Parameter(torch.ones(N)[None,:])
    return cs,log_sigmas,As

def initialize_bernoulli_params_mu(N,x_min,x_max,xs,w):
    # w = new parameter used in the definition of the curve's centroid
    cs = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-w-x_min)/N)[None,0:N])
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*(x_max-x_min)/N)[None,0:N])
    As = torch.nn.Parameter(torch.ones(N)[None,:])
    return cs,log_sigmas,As


def initialize_circbernoulli(N):
    cs = torch.nn.Parameter(torch.arange(0,2*np.pi,2*np.pi/N)[None,0:N])
    sigmas = (torch.ones(N)*2*np.pi/(2*N))[None,:]
    log_ks = torch.nn.Parameter(torch.log(1/sigmas))
    #As = torch.nn.Parameter(torch.exp(-torch.exp(log_ks))*torch.ones(N)[None,:])
    As = torch.nn.Parameter(torch.ones(N)[None,:])
    return cs,log_ks,As

#ENCODER DEFINITION  
class CategoricalEncoder(torch.nn.Module):
    #It returns for each stimulus x a vector of (unnormalized) probabilities 
    # (i.e.logits) of activation for each neuron, which is a quadratic function
    # of x. 
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
    # Encoder returning for N neurons their unnormalized probabilities of being active (i.e. logits),as 
    # a quadratic function of x
    def __init__(self,N,x_min,x_max,xs):
        super().__init__()
        self.cs, self.log_sigmas,self.As  = initialize_bernoulli_params(N,x_min,x_max,xs)
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

class BernoulliEncoderLinPars(torch.nn.Module):
    # Encoder returning for N neurons their unnormalized probabilities of being active (i.e. logits),as 
    # a quadratic function of x
    #Same as BernoulliEncoder, but different parametrization of the parameters: alfa, beta, gamma
    def __init__(self,N,x_min,x_max,xs):
        super().__init__()
        cs, log_sigmas,As  = initialize_bernoulli_params(N,x_min,x_max,xs)
        inv_sigmas = 0.5*torch.exp(-2*log_sigmas)
        self.logalpha = torch.nn.Parameter(torch.log(inv_sigmas))
        self.beta = torch.nn.Parameter(2*cs*inv_sigmas)
        self.gamma = torch.nn.Parameter(- (cs**2)*inv_sigmas + torch.log(As))
    def forward(self,x):
        # x has shape [bsize_dim,x_dim], c,log_sigma,A has shape [x_dim, N]
        eta = -(x**2)@torch.exp(self.logalpha) +  x@self.beta + self.gamma
        
        return eta
    def sample(self,x,nsamples):
        p_r_x = torch.distributions.bernoulli.Bernoulli(logits = self.forward(x))
        r = p_r_x.sample((nsamples,)).transpose(0,1)
        return r

class CircularBernoulliEncoder(torch.nn.Module):
    # Encoder returning for N neurons their unnormalized probabilities of being active (i.e. logits),as 
    # a quadratic function of x
    
    def __init__(self,N):
        super().__init__()
        self.cs, self.log_ks,self.As  = initialize_circbernoulli(N)
    def forward(self,theta): 
        etas = torch.exp((self.log_ks)*torch.cos(theta - self.cs)) + torch.log(self.As)
        return etas
    def sample(self,theta,nsamples):
        p_r_x = torch.distributions.bernoulli.Bernoulli(logits = self.forward(theta))
        r = p_r_x.sample((nsamples,)).transpose(0,1)
        return r



# %%
## Decoders
# After a sampling from the latent space, we have to get a matrix [bsize,n_samples,N] of neural activation.
# The decoder may have different forms, but should return the parameters of an output distribution.

#DECODER INITIALIZATION
def initialize_MOG_params(N,x_min,x_max, x):
    #Initialize parameters arranging centers equally spaced in the range x_min x_max,
    # and the width as 5 times the spacing between centers

    #Need to perform KMEANS to initialize the mus
    kmeans = cluster.KMeans(n_clusters=N, init='random',
                        n_init=10, max_iter=10, random_state=2)
    C = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_
    mus = torch.nn.Parameter(torch.Tensor(centers))

    #mus = torch.nn.Parameter(torch.arange(x_min,x_max,(x_max-x_min)/N)[:,None])
    log_sigmas = torch.nn.Parameter(torch.log(torch.ones(N)*(x_max-x_min)/N)[:,None])
    qs = torch.nn.Parameter(torch.ones(N)[:,None])
    return qs,mus,log_sigmas

#DECODER DEFINITION
class MoGDecoder(torch.nn.Module):
    #Decoder as a mixture of Gaussians, parameters are q(memberships), mean and variances.
    def __init__(self,N,x_min,x_max,x):
        super().__init__()
        self.qs,self.mus,self.log_sigmas = initialize_MOG_params(N,x_min,x_max,x)
    def forward(self,r):
        # r is a tensor [bsize,lat_samples,N],mus and sigmas are [N,1]. Note that input 
        # should be a one_hot tensor. Alternative is 
        #return self.mus[r],self.log_sigmas[r] #if r are categories
        return r @ self.mus, r @ self.log_sigmas
    def sample(self,r,dec_samples):
        mu_dec,log_sigma_dec = self.forward(r)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.exp(log_sigma_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec    

class GaussianDecoder(torch.nn.Module):
    # Return natural parameters of a gaussian distribution as a linear 
    # combination of neural activities
    def __init__(self,phi0):
        super().__init__()
        self.phi = torch.nn.Parameter(phi0)
        self.b = torch.nn.Parameter(torch.tensor([0,-1e-7]))
        #self.A = torch.nn.Parameter(A0)
    def forward(self,r):
        # r must have shape [bsize,lat_samples,N], phi has shape [N,2].
        # mu and sigma are [bsize,lat_samples]
        eta = r @ self.phi + self.b
        eta1,eta2 = eta[:,:,0],eta[:,:,1]
        mu = -0.5*eta1/eta2
        sigma =  - 0.5/eta2
        return mu, sigma
    def sample(self,r,dec_samples):
        mu_dec,sigma2_dec = self.forward(r)
        #Terrible hack, we should find a way to deal with the rare cases
        # of σ^2 <0
        sigma2_dec[sigma2_dec<0] = torch.sqrt(sigma2_dec[sigma2_dec<0]**2)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.sqrt(sigma2_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec

class MLPDecoder(torch.nn.Module):
    #Multilayer Perceptron with one hidden layer, returning the mean and the 
    #variance of a gaussian distribution
    def __init__(self,N,M):
        super().__init__()
        self.hidden = torch.nn.Linear(N,M)
        self.f = torch.nn.ReLU()
        self.w = torch.nn.Linear(M,2)
    def forward(self,r):
        H = self.f(self.hidden(r))
        mu,log_sigma = torch.split(self.w(H),1,dim=2)
        sigma2 = torch.exp(2*log_sigma)
        return torch.squeeze(mu),torch.squeeze(sigma2)
    def sample(self,r,dec_samples):
        mu_dec,sigma2_dec = self.forward(r)
        #Terrible hack, we should find a way to deal with the rare cases
        # of σ^2 <0
        sigma2_dec[sigma2_dec<0] = torch.sqrt(sigma2_dec[sigma2_dec<0]**2)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.sqrt(sigma2_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec
    
class GaussianDecoder_orig(torch.nn.Module):
    #return mu and sigma of a gaussian distribution
    def __init__(self,mu0,sigma0,q0):
        super().__init__()
        self.q  = torch.nn.Parameter(F.softmax(q0))
        self.mu = torch.nn.Parameter(mu0.transpose(0,1)) #transpose(0,1) a row becomes a colummn
        self.sigma = torch.nn.Parameter(sigma0.transpose(0,1))
    def forward(self,r):
        return self.mu, self.sigma


class MLPDecoder(torch.nn.Module):
    def __init__(self,N,M):
        super().__init__()
        self.hidden = torch.nn.Linear(N,M)
        self.f = torch.nn.ReLU()
        self.w = torch.nn.Linear(M,2)
    def forward(self,r):
        H = self.f(self.hidden(r))
        mu,log_sigma = torch.split(self.w(H),1,dim=2)
        sigma2 = torch.exp(-2*log_sigma)
        return torch.squeeze(mu),torch.squeeze(sigma2)
    def sample(self,r,dec_samples):
        mu_dec,sigma2_dec = self.forward(r)
        #Terrible hack, we should find a way to deal with the rare cases
        # of σ^2 <0
        sigma2_dec[sigma2_dec<0] = torch.sqrt(sigma2_dec[sigma2_dec<0]**2)
        q_x_r = torch.distributions.normal.Normal(mu_dec,torch.sqrt(sigma2_dec))
        x_dec = q_x_r.sample((dec_samples,))
        return x_dec


# %%
