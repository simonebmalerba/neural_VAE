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
# Architecture parameters
N= 10     #Number of neurons
M = 100  #Decoder neurons (D NN)
#Training parameters
N_EPOCHS = 3000
N_SAMPLES =5000
BATCH_SIZE = 100
#lr = 1e-4
#Generate dataset
p_x = torch.distributions.log_normal.LogNormal(1,1) 
x_samples = p_x.sample((N_SAMPLES,))[:,None]     #Points used for model train
x_test = p_x.sample((40000,))[:,None]            #Points used for model test
x_tsorted,_ = x_test.sort(dim=0)
x_sorted,_ = x_samples.sort(dim=0)
x_min,x_max = x_sorted[0,:].item(),x_sorted[-1,:].item()
x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)
x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)
# %%
#Initialize model parameters
#Iterate over different R^*
resume = {}
RtVec = np.linspace(0.3,2.5,num=15)
#RtVec = [0.5,1]
for Rt in RtVec:
    if Rt < 1.3:
        lr = 1e-4
    else:
        lr=1e-3
    print(f"Rate = {Rt}||lr = {lr}")
    enc = BernoulliEncoder(N,x_min-1,x_max+1,x_sorted,w=3)
    dec = MLPDecoder2n(N,M)     #Decoder
    q = rate_ising(N)           #Prior
    q.J.register_hook(lambda grad: grad.fill_diagonal_(0))
    params =   list(enc.parameters()) + list(dec.parameters())  + list(q.parameters())
    opt = torch.optim.Adam(params,lr)
    history = train_Rt(enc,dec,q,x_data,opt,Rt,N_EPOCHS = N_EPOCHS,lr_b = 0.1)
    resume[Rt] = {'encode' :enc.state_dict(),
                'decoder' : dec.state_dict(),
                'q'      : q.state_dict(),
                'history' : history,
                'x_test'  : x_tsorted
    }
PATH = os.getcwd() + "/data/LN_prior_N=10_q=Ising_#4.pt"
torch.save(resume, PATH)
# %%

# %%
#torch.load(PATH)
# %%
