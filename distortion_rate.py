#%%
from re import A
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import pickle
import itertools
import random
import os
from src.encoders_decoders import *
from src.losses import *
from src.useful_functions import *
from torch.utils.data import DataLoader
from joblib import Parallel,delayed
# This script contains the main training loops.
# It saves a dictionary of N_trials models trained with different target rates, together with the training history
# The saved models can then be analyzed.
# %%
def train_Rt(enc,dec,q,x_data,x_test,opt,Rt,N_EPOCHS=500,lr_b = 0.1):
    # Train parameters of VAE (enc + dec) stored in opt at a given target rate
    # by minimizing -ELBO (D+R).
    history = { "loss" : [],
                "distortion" : [],
                "rate" : [],
                "beta" : [1] ,
                "distortion_test" : [],
                "rate_test" : []}

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
        if not e % 100:
            # Test decoder on larger set of stimuli
            history["distortion_test"].append(distortion_analytical_linear(x_test,enc,dec,q.r_all).mean().item())
            history["rate_test"].append(q(enc,x_test).mean().item())
        print(f'Epoch: {e} ||Rate: {history["rate"][-1]}||',
            f'ELBO:{history["loss"][-1]}||',
            f'Distortion: {history["distortion"][-1]}||Beta = {history["beta"][-1]}')
    history["beta"].pop()
    return history

def vary_R(RtVec,x_test):
    # Train a set of VAE for different value of the target rate in RtVec.
    resume = {}
    # Sample train and test data from the prior over stimuli.
    x_samples = p_x.sample((N_SAMPLES,))[:,None]
    x_sorted,_ = x_samples.sort(dim=0)
    x_min,x_max = x_sorted[0,:].item(),x_sorted[-1,:].item()
    x_data = torch.utils.data.DataLoader(x_samples,batch_size=BATCH_SIZE)
    
    for Rt in RtVec:
        #Model might become unstable for low values of the distortion
        if Rt < 1.5:
            lr = 1e-4
        else:
            lr=1e-3
        print(f"Rate = {Rt}||lr = {lr}")
        # Initialize encoder, decoder and prior parameters
        enc = BernoulliEncoder(N,x_min-1,x_max+1,x_sorted,w=2)
        dec = MLPDecoder(N,M)     
        q = rate_ising(N)          
        q.J.register_hook(lambda grad: grad.fill_diagonal_(0))
        params =   list(enc.parameters()) + list(dec.parameters())  + list(q.parameters())
        opt = torch.optim.Adam(params,lr)
        history = train_Rt(enc,dec,q,x_data,x_test,opt,Rt,N_EPOCHS = N_EPOCHS,lr_b = 0.1)
        resume[Rt] = {'encode' :enc.state_dict(),
                    'decoder' : dec.state_dict(),
                    'q'      : q.state_dict(),
                    'history' : history,
                    'lr'      :lr
        }
    return resume
#%%
N_SAMPLESVec= [50,100,250,500,1000,2000]
BATCH_SIZEVec= [8,16,16,32,128,128]
N_EPOCHSVec = [7000,7000,7000,6000,4000,4000]
for i in range(7):
    # Population parameters
    N = 12       # Number of encoding neurons
    M = 100     # DNN hidden layer size
    # Training hyperparameters
    N_EPOCHS = N_EPOCHSVec[i]
    N_SAMPLES = N_SAMPLESVec[i]
    BATCH_SIZE = BATCH_SIZEVec[i]
    N_TRIALS = 16 #Different initializations and data samples.
    # Stimulus prior distribution
    #p_x = torch.distributions.log_normal.LogNormal(1,1)
    w = torch.distributions.Categorical(torch.tensor([0.3,0.2,0.5]))
    gs = torch.distributions.normal.Normal(torch.Tensor([-4,0,2]),torch.tensor([1,0.5,1]))
    p_x = torch.distributions.mixture_same_family.MixtureSameFamily(w,gs)
    x_test,_ = (p_x.sample((3000,))[:,None]).sort(dim=0)
    #x_test = torch.utils.data.DataLoader(x_test,batch_size=BATCH_SIZE)
    # Vector of target rates
    RtVec = np.linspace(0.2,2.6,num=10)
    r_list = Parallel(n_jobs=-1)(delayed(vary_R)(RtVec,x_test) for n in range(N_TRIALS))

    PATH = os.getcwd() + f"/data/MOG_prior_samples={N_SAMPLES}_N=12_q=Ising_lrs=15.pt"

    torch.save(r_list, PATH)
# %%
