#%%
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import cluster
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

#%%
def save_model():
    return


#%%
def encoder_plots(encoder,x_fine,lat_samp = 30):
    with torch.no_grad():
        #Plot tuning curves and mean number of spikes of encoder
        r = encoder.sample(x_fine,lat_samp)
        #mu_dec,log_sigma_dec = decoder(r)
        fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(10,5))
        axs[0].plot(x_fine,torch.sigmoid(encoder(x_fine)).detach())
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p(r=1|x)')
        axs[1].plot(x_fine,r.sum(dim=2).mean(dim=1))
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('# spikes')
    return fig,axs

def decoder_plots(encoder,decoder,x_fine,lat_samp = 30):
    #Plot mean square error and mean and variance of the decoder
    with torch.no_grad():
        r = encoder.sample(x_fine,lat_samp)
        mu_dec,log_sigma_dec = decoder(r)
        x_ext = decoder.sample(r,lat_samp)
        mseVec = ((x_ext - x_fine[None,:])**2).mean(dim=(0,2))
        meanMSE = mseVec.mean()
        fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(15,5))
        axs[0].plot(x_fine,mu_dec.mean(dim=1))
        axs[0].plot(x_fine,x_fine,label="y=x")
        axs[0].legend()
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('μ')
        axs[1].plot(x_fine,torch.exp(log_sigma_dec).mean(dim=1))
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('σ')
        axs[2].plot(x_fine,mseVec)
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('MSE')
        axs[2].set_title(f'MSE: {meanMSE:.2f}')
        axs[2].plot()
    return fig,axs
##
def training_plots(loss,distortion,rate,beta):
    L = len(loss)
    minLoss = min(loss)
    emin = loss.index(minLoss)
    fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(15,5))
    axs[0].plot(loss,label='ELBO')
    axs[0].plot(distortion,label='Distortion')
    axs[0].plot(rate,label='Rate')
    axs[0].legend()
    axs[0].annotate(f'MinElbo: {minLoss:.2f}',(emin,minLoss),
        fontsize=14,textcoords = 'axes fraction',xytext=(0.5,0.5),
        arrowprops=dict(facecolor='black', shrink=0.05))
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Losses")
    axs[1].scatter(rate,distortion,c =range(L), cmap="viridis")
    axs[1].set_xlabel("Rate")
    axs[1].set_ylabel("Distortion")
    axs[2].scatter(range(L),beta,c =range(L), cmap="viridis")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Beta")
    return fig,axs

def generative_model_analytical_plots(q,decoder,p_x,x_fine,indices=None):
    _,N = q.h.shape
    if indices == None:
        indices= range(N)
    with torch.no_grad():
        fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(15,5))
        mu_dec_all,log_sigma_all = decoder(q.r_all.transpose(0,1)[:,None,:])
        q_r = torch.softmax(q.h@q.r_all + (q.r_all*(q.J@q.r_all)).sum(dim=0),1)
        wq = torch.distributions.Categorical(q_r)
        gsq = torch.distributions.normal.Normal(mu_dec_all,torch.exp(log_sigma_all))
        q_x = torch.distributions.mixture_same_family.MixtureSameFamily(wq,gsq)
        axs[0].plot(x_fine,10**q_x.log_prob(x_fine),label= "q(x)")
        axs[0].plot(x_fine,10**p_x.log_prob(x_fine),label= "p(x)")
        axs[0].legend()
        axs[1].imshow(q.J[:,indices][indices,:])
        axs[1].set_title('J')
        axs[2].scatter(range(N),q.h[:,indices])
        axs[2].set_title('h')
    return fig,axs
# %%
