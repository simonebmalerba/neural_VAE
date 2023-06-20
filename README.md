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
Distortion functions measure the reconstruction of a stimulus (likelihood) after it has been encoded into a neural activity pattern and then decoded.
Rate functions measure the kullback-leibler divergence between the encoding distribution and a prior distribution over neural activity patterns.

The scripts distortion_rate.py and non-analytical_prior.py constitute an example of training procedure: different models for different values of the maximum rate allowed (target rate) are trained and saved, and can be succesively analyzed with custom scripts. An example of simulations result can be found in the data folder (compressed).
Two notebooks contains typical analyses that can be run to study the behavior of the model for different values of the target rate and a single instance of the model.



