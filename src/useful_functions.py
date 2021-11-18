#%%
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import cluster
import pandas as pd
import matplotlib.pyplot as plt
#%%

def show_plots(elements):
    
    combined_data = torch.stack(elements).detach().numpy()
    #Get the min and max of all your data
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    fig = plt.figure(figsize=(5,5))
    for i in range(len(elements)):
        ax = fig.add_subplot(len(elements), 1, i+1)
        #Add the vmin and vmax arguments to set the color scale
        im=ax.imshow(elements[i],cmap=plt.cm.YlGn, vmin = _min, vmax = _max)
        ax.set_title("J")
        fig.colorbar(im, ax=ax)
        #ax.autoscale(True)

    plt.show()
#%%
def encoder_plots(encoder,x_fine,lat_samp = 30):
        #Plot tuning curves and mean number of spikes of encoder
        r = encoder.sample(x_fine,lat_samp)
        #mu_dec,log_sigma_dec = decoder(r)
        fig,axs = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
        axs[0].plot(x_fine,torch.sigmoid(encoder(x_fine)).detach())
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p(r=1|x)')
        axs[1].plot(x_fine,r.sum(dim=2).mean(dim=1))
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('# spikes')

def decoder_plots(encoder,decoder,x_fine,lat_samp = 30):
    #Plot mean square error and mean and variance of the decoder
    with torch.no_grad():
        r = encoder.sample(x_fine,lat_samp)
        mu_dec,log_sigma_dec = decoder(r)
        x_ext = decoder.sample(r,10)
        mseVec = ((x_ext - x_fine[None,:])**2).mean(dim=(0,2))
        fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(10,10))
        axs[0,0].plot(x_fine,mu_dec.mean(dim=1))
        axs[0,0].plot(x_fine,x_fine,label="y=x")
        axs[0,0].legend()
        axs[0,0].set_xlabel('x')
        axs[0,0].set_ylabel('μ')
        axs[1,0].plot(x_fine,torch.exp(log_sigma_dec).mean(dim=1))
        axs[1,0].set_xlabel('x')
        axs[1,0].set_ylabel('σ')
        axs[0,1].plot(x_fine,mseVec)
        axs[0,1].set_xlabel('x')
        axs[0,1].set_ylabel('MSE')
        axs[1,1].plot()