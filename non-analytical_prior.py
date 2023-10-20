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
from joblib import Parallel,delayed
# Same as distortion_rate.py, but for a non-analytical distribution over stimuli.
# %%
def train_Rt(enc,dec,q,x_data,opt,Rt,N_EPOCHS=500,lr_b = 0.1):
    # Train parameters of VAE (enc + dec) stored in opt at a given target rate
    # by minimizing -ELBO (D+R).
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

def vary_R(RtVec,x_data):
     # Train a set of VAE for different value of the target rate in RtVec.
    resume = {}
    # Sample data from the prior over stimuli.
    for Rt in RtVec:
        if Rt < 1.7:
            lr = 1e-4
        else:
            lr=1e-3
        print(f"Rate = {Rt}||lr = {lr}")
        x_samples = fs[d.sample((N_SAMPLES,))[:,None]]
        x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)
        # Initialize encoder, decoder and prior parameters
        enc = BernoulliEncoder(N,x_min-1,x_max+1,x_sorted,w=0.7)
        dec = MLPDecoder(N,M)     
        q = rate_ising(N)          
        q.J.register_hook(lambda grad: grad.fill_diagonal_(0))
        params =   list(enc.parameters()) + list(dec.parameters())  + list(q.parameters())
        opt = torch.optim.Adam(params,lr)
        history = train_Rt(enc,dec,q,x_data,opt,Rt,N_EPOCHS = N_EPOCHS,lr_b = 0.1)
        resume[Rt] = {'encode' :enc.state_dict(),
                    'decoder' : dec.state_dict(),
                    'q'      : q.state_dict(),
                    'history' : history,
                    'lr'      :lr
        }
    return resume
#%%
# Population parameters
N = 12
M = 100
# Training hyperparameters
N_EPOCHS = 6000
N_SAMPLES =8000
lr = 1e-2
BATCH_SIZE = 256
N_TRIALS = 16
# Definition of power-law probability density function (from Ganguli&SImoncelli,2016)
f0 = 1.52 #0#
p = 2.61 #0.84#
A = 2.4e6/10**(3*p) #0.06#
density = lambda f :  A/(f0**p + f**p)
f_bin = torch.logspace(-1.11, 1.6,steps=1001) # Create bin edges for histogram
Df = torch.diff(f_bin)
fs = f_bin[0:-1] + Df/2
pf = density(fs)
Z = (Df*pf).sum().item() # Normalize to obtain pdf
d = torch.distributions.categorical.Categorical(probs = pf*Df)
# Sample data from probabiltiy distribution
x_samples = fs[d.sample((N_SAMPLES,))[:,None]]
x_test = fs[d.sample((10000,))[:,None]]
x_tsorted,_ = x_test.sort(dim=0)
x_sorted,indices = x_samples.sort(dim=0)
x_min,x_max = x_sorted[0,:].item(),x_sorted[-1,:].item()
x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)

RtVec = np.linspace(0.4,2.7,num=8)

r_list = Parallel(n_jobs=16,prefer='threads')(delayed(vary_R)(RtVec,x_data) for n in range(N_TRIALS))

PATH = os.getcwd() + "/data/freq_dist_N=12_q=Ising_lrs=1_7_highf2_narrowinit.pt"
torch.save(r_list, PATH)
