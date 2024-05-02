# Jointly Efficient Encoding and Decoding in Neural Populations.
## Introduction

This repository contains the custom code associated with the manuscript titled "Jointly Efficient Encoding and Decoding in Neural Populations." (Blanco Malerba et al., 2023).
The manuscript investigates a novel approach for efficient encoding and decoding of neural activity in populations of neurons using a Variational Autoencoder framework.

## Dependencies

The code runs with python 3.9 and the following requirements:
* numpy==1.25.0
* torch==2.0.1
* scikit-learn==1.2.2
For some analyses of the results, the following packages are needed
* scipy==1.9.3
* KDEpy==1.1.0
* matplotlib==3.6.2

## Brief Description

The src folder contains model definitions (encoder_decoders.py) and loss functions (losses.py).
An encoder maps a stimulus, x, to a vector of neural responses, r, a decoder maps a vector of neural responses, r, to a probabiltiy distribution over stimuli.
We consider binary neurons, whose response is 0 or 1.
The encoder consists of a population of neurons whose selectivity is parametrized with bell-shaped tuning curves, while the decoder is a generic parametric function (e.g., a deep neural network). 
The mathematical description of the model can be found in Blanco Malerba et al., 2023

The parameters of the encoder and decoder are trained so as to optimize a loss function; the file losses.py allow to define the different contributions to the loss function.
In a Variational Autoencoder framework, the loss function is the Evidence Lower Bound (ELBO), which is the sum of two contributions: distortion and rate.
Distortion function measure the reconstruction of a stimulus (likelihood) after it has been encoded into a neural activity pattern and then decoded.
Rate function measure the kullback-leibler divergence between the encoding distribution and a prior distribution over neural activity patterns.

The scripts distortion_rate.py and non-analytical_prior.py provide an example of training procedure: different models for different values of the maximum rate allowed (target rate) are trained and saved, and can be succesively analyzed with custom scripts.

The experimental results for the frequency differency limens, obtained from Moore BCJ, (1979), are present in the data folder as csv files.

## Figures details

The file data_PLOS.zip contains data to reproduce figures of the paper.
Simulations files (trained models for different values of the target rate) are in the folder 'Models', notebooks to analyze simulations and plot results are in the 'Notebook' folder, data plotted in the figures are in the folder 'Figures_PLOS' as .xlsx files. 
Below, we report the association between files and notebooks for each figure:

- Fig2, Fig S1, Fig S6: File: 'LN_prior_samples=2000_N=12_q=Ising_lrs=15.pt'
                        Notebook: 'd_r_analysis_singleR_LNprior_example.ipynb'

- Fig3, Fig S7:  File: 'N_prior_N=12_q=Ising_lrs=1_5.pt'
                 Notebook: 'd_r_analysis_vsR_Gaussian_example.ipynb'

- Fig4, Fig7, and Fig8: File: 'LN_prior_samples=2000_N=12_q=Ising_lrs=15.pt'
                        Notebook: 'd_r_analysis_vsR_LN_example.ipynb'
- Fig5: Files: 'LN_prior_samples=2000_N=*.pt'
        Notebook: 'd_r_analysis_vsR_N=_example.ipynb'

- Fig6: Files: 'LN_prior_samples=*.pt'
        Notebook: 'd_r_analysis_vsR_samples=_LN_example.ipynb'

- Fig9:   Files: 'freq_dist_dec=LN_N=12_q=Ising_lrs=1_5.pt', 'freq_dist_N=12_q=Ising_lrs=1_5.pt' and csv files 'subj_*.csv'
          Notebook: 'd_r_analysis_vsR_freq_example.ipynb'

- FigS2 and Fig S8: File:'LN_prior_dec=LN_samples=2000_N=12_q=Ising_lrs=15.pt'
                    Notebook: 'd_r_analysis_vsR_LNdec_example.ipynb'
- FigS3: Files: 'LN_prior_samples=1000_N=12_q=Bernoulli_lrs=15.pt'
         Notebook: 'd_r_analysis_vsR_bernoulli_example.ipynb'

- FigS4: File: 'MOG_prior_samples=2000_N=12_q=Ising_lrs=15.pt'
         Notebook: 'd_r_analysis_vsR_MoG_example.ipynb'

- FigS5: Files:  'MOG_prior_samples=*.pt'
         Notebook: 'd_r_analysis_vsR_samples=_MoG_example.ipynb


