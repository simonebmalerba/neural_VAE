#%%
import torch
import torch.nn.functional as F
import numpy as np
import numpy
from sklearn import cluster
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import scipy.stats
import scipy
import random
from src.encoders_decoders import *
from src.losses import *
#%%

#PLOTS ENCODING
def plot_enc(enc, x):
    plt.plot(x,torch.sigmoid(enc(x)).detach())
    plt.xlabel("Stimulus")
    plt.ylabel("Encoding")


def plot_r(enc, x):
    r = enc.sample(x,50)
    plt.plot(x, r.sum(dim=2).mean(dim=1) )
    plt.ylabel("r")
    plt.xlabel("Stimulus")


#PLOTS DECODING
def plot_var_dec_circ(enc, dec, x):
    r = enc.sample(x,50)
    mu_dec, log_k_dec = dec(r)

    with torch.no_grad():
       plt.plot(x,torch.exp(log_k_dec).mean(dim=1).detach())
       plt.ylabel("var")
       plt.xlabel("Stimulus")
       plt.xlim(-math.pi,math.pi)

def plot_var_dec_lin(enc, dec, x):
    r = enc.sample(x,50)
    mu_dec, sigma_dec = dec(r)

    with torch.no_grad():
       plt.plot(x,sigma_dec.mean(dim=1).detach())
       plt.ylabel("var")
       plt.xlabel("Stimulus")

def plot_mu_dec_circ(enc, dec, x):
    r = enc.sample(x,50)
    mu_dec, log_k_dec = dec(r)

    with torch.no_grad():
       plt.plot(x,mu_dec.mean(dim=1).detach())
       plt.ylabel("mu")
       plt.xlabel("Stimulus")
       plt.xlim(-math.pi,math.pi)

def plot_mu_dec_lin(enc, dec, x):
    r = enc.sample(x,50)
    mu_dec, sigma_dec = dec(r)

    with torch.no_grad():
       plt.plot(x,mu_dec.mean(dim=1).detach())
       plt.ylabel("mu")
       plt.xlabel("Stimulus")
       


#PLOT LOSSES
def plot_loss(loss, distortion, rate):
    plt.plot(loss,'g', label = 'loss')
    plt.plot(distortion, 'r', label = 'distortion')
    plt.plot(rate, 'y', label = 'rate')
    plt.legend()
    plt.show()


#PLOT MSE
def plot_mse_circ(enc, dec, x):
    r = enc.sample(x,50)
    x_dec = dec.sample(r,100)
    mseVec = 2 - 2*torch.cos((x_dec - x[None,:])).mean(dim=(0,2))

    plt.plot(x,mseVec.detach()/20)
    plt.ylabel("MSE")
    plt.xlabel("Stimulus")

def plot_mse_lin(enc,dec,x):
    r = enc.sample(x,50)
    x_dec = dec.sample(r,100)
    mseVec = ((x_dec - x[None,:])**2).mean(dim=(0,2))
    plt.plot(x,mseVec.detach()/20)
    plt.ylabel("MSE")
    plt.xlabel("Stimulus")


#PLOT PDF

def plot_pdf_circ(x):
    kde = scipy.stats.gaussian_kde(x[:,0])
    t_range = np.linspace(-5,5,200)
    plt.plot(t_range,kde(t_range))
    plt.xlabel("stimulus")
    plt.ylabel("pdf")

def plot_pdf_lin(x):
    kde = scipy.stats.gaussian_kde(x[:,0])
    t_range = np.linspace(-2,15,200)
    plt.plot(t_range,kde(t_range))
    plt.xlabel("stimulus")
    plt.ylabel("pdf")


#PLOT J MATRICES  
def show_plots(elements):
    
    combined_data = torch.stack(elements).detach().numpy()
    #Get the min and max of all your data
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    fig = plt.figure(figsize=(10,10))
    for i in range(len(elements)):
        ax = fig.add_subplot(len(elements), 1, i+1)
        #Add the vmin and vmax arguments to set the color scale
        im=ax.imshow(elements[i],cmap=plt.cm.YlGn, vmin = _min, vmax = _max)
        ax.set_title("J")
        fig.colorbar(im, ax=ax)
        #ax.autoscale(True)

    plt.show()


#GENERATIVE MODEL TEST

def generative_model_circ(rate, dec):
    #function that returns the reconstructed x

    #prior params
    prior_params = list(rate.parameters())
    h_learn = prior_params[0]# 1 x 10
    J_learn = prior_params[1] # 10 x 10
    
    #space of all possible r
    N = 10
    r_all = np.asarray(list(itertools.product([0, 1], repeat=N)))
    r_all = r_all.transpose()
    
    #computing of q(r)
    expo = torch.exp(torch.as_tensor(h_learn.detach().numpy()@r_all + np.sum(r_all*(J_learn.detach().numpy()@r_all), axis = 0)))
    Z = expo.sum(dim=1)
    q_r = torch.squeeze( (1/Z)*expo)
    
    #dec parameters
    r_all1 = np.asarray(list(itertools.product([0, 1], repeat=N)))
    r_all = torch.as_tensor(r_all1[:,None,:], dtype = torch.float32)
    mu_dec_all,log_k_dec_all = dec(r_all)
    
    #computing of q(x) (mixture of Von Mises weighted by q(r))
    weights = q_r
    means = mu_dec_all
    stdevs = torch.exp(log_k_dec_all)
    mix = torch.distributions.categorical.Categorical(weights)
    compo = torch.distributions.independent.Independent(torch.distributions.von_mises.VonMises(means, stdevs),0)
    q_x = torch.distributions.mixture_same_family.MixtureSameFamily(mix, compo)
    
    #sampling from q(x)
    N_SAMPLES=5000
    x_samples = q_x.sample((N_SAMPLES,))[:,None]
    x_samples = torch.squeeze(x_samples)

    return x_samples



def generative_model_lin(rate, dec):
    #function that returns the reconstructed x

    #prior params
    prior_params = list(rate.parameters())
    h_learn = prior_params[0]# 1 x 10
    J_learn = prior_params[1] # 10 x 10
    
    #space of all possible r
    N = 10
    r_all = np.asarray(list(itertools.product([0, 1], repeat=N)))
    r_all = r_all.transpose()
    
    #computing of q(r)
    expo = torch.exp(torch.as_tensor(h_learn.detach().numpy()@r_all + np.sum(r_all*(J_learn.detach().numpy()@r_all), axis = 0)))
    Z = expo.sum(dim=1)
    q_r = torch.squeeze( (1/Z)*expo)
    
    #dec parameters
    r_all1 = np.asarray(list(itertools.product([0, 1], repeat=N)))
    r_all = torch.as_tensor(r_all1[:,None,:], dtype = torch.float32)
    mu_dec_all,sigma_dec_all = dec(r_all)
    
    #computing of q(x) (mixture of Von Mises weighted by q(r))
    weights = q_r
    means = mu_dec_all
    stdevs = sigma_dec_all
    mix = torch.distributions.categorical.Categorical(weights)
    compo = torch.distributions.independent.Independent(torch.distributions.normal.Normal(mu_dec_all,sigma_dec_all),0)
    q_x = torch.distributions.mixture_same_family.MixtureSameFamily(mix, compo)
    
    #sampling from q(x)
    N_SAMPLES=5000
    x_samples = q_x.sample((N_SAMPLES,))[:,None]
    x_samples = torch.squeeze(x_samples)

    return x_samples
#%%
