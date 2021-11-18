import torch
import matplotlib.pyplot as plt


def draw_hist(hist):
        fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(10, 10));
        axs[0,0].plot(hist['ELBO']);
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('ELBO')
        axs[0,1].plot(hist['D']);
        axs[0,1].set_ylim(0,30)
        axs[0,1].set_ylabel('Distortion')
        axs[0,1].set_xlabel('Epochs')
        axs[1,1].plot(hist['R']);
        axs[1,1].set_ylabel('Rate')
        axs[1,1].set_xlabel('Epochs')
        axs[1,0].plot(hist['MSE']);
        axs[1,0].set_ylabel('MSE')
        axs[1,0].set_xlabel('Epochs')
        plt.show()
        return fig,axs    
def encoder_properties(encoder,x_fine):
        fig,axs = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
        axs[0].plot(x_fine,torch.sigmoid(encoder(x_fine)).detach())
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p_j(x)')
        axs[1].scatter(encoder.cs.detach(),torch.exp(encoder.log_sigmas).detach())
        axs[1].set_xlabel('c_j')
        axs[1].set_ylabel('sigma_j')
        plt.show()

def decoder_meanvar(encoder,decoder,x_fine,lat_samp = 10):
        #
        r = encoder.sample(x_fine,lat_samp)
        mu_dec,log_sigma_dec = decoder(r)
        fig,axs = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
        axs[0].plot(x_fine,mu_dec.mean(dim=1).detach())
        axs[0].plot(x_fine,x_fine,label="y=x")
        axs[0].legend()
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('μ')
        axs[1].plot(x_fine,torch.exp(log_sigma_dec).mean(dim=1).detach())
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('σ')


def errors_analysis(encoder,decoder,x_fine):
        r = encoder.sample(x_fine,10)
        x_dec = decoder.sample(r,10)
        mseVec = ((x_dec - x_fine[None,:])**2).mean(dim=(0,2))
        print("MSE = ",mseVec.mean().item())
        fig,axs  = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
        axs[0].plot(x_fine,mseVec.detach())
        axs[0].set_yscale("log")
        axs[0].set_xlabel('x')
        axs[0].set_ylabel(r'$\Delta x$')
        errors = np.abs(torch.flatten((x_dec - x_fine[None,:])))
        axs[1].hist(errors.numpy(),log=True)
        axs[1].set_xlabel(r'$\Delta x$')
        axs[1].set_ylabel(r'$p(\Delta x)$')
        plt.show()