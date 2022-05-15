#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
import itertools
import random
import os
from src.encoders_decoders import *
from src.losses import *
from src.useful_functions import *
from torch.utils.data import DataLoader
# %%
def train_Rt(enc,dec,q,x_data,opt,Rt,N_EPOCHS=500,lr_b = 0.1):
    #Train parameters of VAE specified in `opt`
    history = { "loss" : [],
                "distortion" : [],
                "rate" : [],
                "beta" : [1]   }

    for e in range(N_EPOCHS):
        lav = dav = rav = 0
        beta = history["beta"][-1]
        for x_ in x_data:
            rate = q(enc,x_)
            distortion = distortion_analytical_linear(x_,enc,dec,q.r_all)
            loss =  distortion +beta*rate
            opt.zero_grad()
            loss.backward()
            opt.step()
            lav += distortion + rate
            dav += distortion
            rav += rate
            if torch.isnan(loss):
                break;
        history["loss"].append(lav.item()/len(x_data))
        history["rate"].append(rav.item()/len(x_data))
        history["distortion"].append(dav.item()/len(x_data))
        #Update constraint
        beta += lr_b*(history["rate"][-1]-Rt)
        beta = beta if beta>0 else 0
        history["beta"].append(beta)
        print(f'Epoch: {e} ||Rate: {history["rate"][-1]}||',
            f'ELBO:{history["loss"][-1]}||',
            f'Distortion: {history["distortion"][-1]}||Beta = {history["beta"][-1]}')
    history["beta"].pop()
    return history
#%%
#Architecture parameters and distributions of stimuli
N = 10
K = 5
M = 100
#Training parameters.
#PRE_EPOCHS = 100
N_EPOCHS = 3000
N_SAMPLES =10000
lr = 1e-2
BATCH_SIZE = 200
#define manually pdf
f0 = 1.52 #0#
p = 2.61 #0.84#
A = 2.4e6/10**(3*p) #0.06#
density = lambda f :  A/(f0**p + f**p)
#create bin edges for histogram
f_bin = torch.logspace(-1, 1,steps=501)
Df = torch.diff(f_bin)
fs = f_bin[0:-1] + Df/2
pf = density(fs)
Z = (Df*pf).sum().item() #Normalize to obtain pdf
#pf /= pf.sum()
#plt.scatter(fs,20*torch.log10(pf))
#plt.xscale("log")
d = torch.distributions.categorical.Categorical(probs = pf)
x_samples = fs[d.sample((N_SAMPLES,))[:,None]]
x_test = fs[d.sample((40000,))[:,None]]
x_tsorted,_ = x_test.sort(dim=0)
x_sorted,indices = x_samples.sort(dim=0)
x_min,x_max = x_sorted[0,:].item(),x_sorted[-1,:].item()
x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)