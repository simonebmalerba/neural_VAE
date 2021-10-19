#!/usr/bin/env python
#coding=utf-8

'''Extracts spike trains in the desired state, runs Ising inference and 
produces figures from model/data comparison

to compile .c code for python from terminal and generate .so files
(need to re-do when transitioning from 32bit to 64bit machine and vice versa):
    g++ -shared -Wl,-soname,MCspins -o MCspins.so -fPIC MCspins.c
for example file MCspins.c

22/09/2016 T.A. Nghiem trang-anh.nghiem@cantab.net
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy import stats
from scipy import fftpack
from numpy import linalg as LA
from infer_pK_Kt_bytype import optimalLearningParallel

import ctypes

MCspins = ctypes.CDLL('./MCspins.so')


bin_per_ms = 30.0

bin_dur = 50.0 #ms

n_jitter = 1 #1000
stddev_jitter = 10 

shift_min = -30
shift_max = 30
shift_step = 1

bin_exclude = 10.0*60*1000*bin_per_ms*10/bin_dur 
sub_train = 300

half_peak_width = 6 #for central peal peak removal in autocorrelogram



# plot configuration
#cm_to_inch = 0.393701
column_width = 7
figsize_rect = np.dot((1.,2./3.), column_width)
figsize_square = np.dot((1.,1.), column_width)
marker_size = 4
font_size = 15
line_width = 2.
##

def filter_spikes_by_periods(binned, periods):
    # from compare_states.py, given by Bartosz
    times = np.arange(len(binned))*bin_dur #np.dot(np.arange(len(binned)),bin_dur)

    # get start/end of periods
    start_times = periods['start time (ms)']
    finish_times = periods['finish time (ms)']

    # iterate over periods
    filtered_spikes = []
    for start, finish in zip(start_times, finish_times):
        # find bin indices where the period starts/ends
        idx_start, idx_finish = np.searchsorted(times, [start, finish])
        filtered_spikes.append(binned[idx_start:idx_finish])
    # concatenate periods together
    # this will produce discontinous time!
    filtered_spikes = np.concatenate(filtered_spikes)# , axis=1)

    return filtered_spikes

def fano_factor(binned_arr):
    return np.var(binned_arr) / np.mean(binned_arr)

def spike_train(t_spikes, bin_dur):
    train = np.zeros(int(np.round(t_spikes[-1]/bin_dur*1.0)) + 1, 
                     dtype=np.uint8)
    for t_sp in t_spikes:
        bin_number = int(np.floor(t_sp/bin_dur*1.0))
#        train[bin_number] += 1
        train[bin_number] = 1 # max 1 spike per bin
    return train        
    
def jitter(t_spikes, stddev):
    t_jittered = stddev*np.random.randn(len(t_spikes)) + t_spikes
#    plt.plot(t_spikes, t_jittered, 'k')
#    plt.show()
    return t_jittered

def spike_train_length_equalize(train_test, train_ref):
    # ensuring that spike trains are the same dimension
    if len(train_ref) < len(train_test):
        train_test =  train_test[:len(train_ref)]
    elif len(train_test) < len(train_ref):
        train_ref =  train_ref[:len(train_test)]
    return (train_ref, train_test)

def spike_train_shift(train_test, n_bins_shift):
    if n_bins_shift >= 0:
        train_shift = train_test[n_bins_shift:] + [0]*n_bins_shift
    elif n_bins_shift < 0:
        train_shift = [0]*(-n_bins_shift) \
            + train_test[:(len(train_test) + n_bins_shift)]
    return train_shift

def correlation_per_trial(train_ref, train_test):
    # computing the correlation
    corr = np.dot(train_ref,train_test)
    return corr

def correlation_pearson(train_ref, train_test): 
    corr = stats.pearsonr(train_ref, train_test)
    return corr[0]

def filter_centre(half_peak_width):
    filt = np.ones(sub_train)
    half_train = int(np.floor(sub_train/2.0))
    filt[half_train - half_peak_width:half_train + half_peak_width] = \
        np.zeros(2*half_peak_width)
    return filt

def correlation_fft(train_ref, train_test):
    i_tr = 0
    fft_corr = []
    while i_tr*sub_train <= len(train_ref):            
        # divide spike trains into sub-trains
        train_min = sub_train*i_tr
        if sub_train*(i_tr+1) > len(train_ref): 
            train_max = len(train_ref) - 1
        else:    
            train_max = sub_train*(i_tr+1)
        # take Fourier transform
        if len(train_ref[train_min:train_max]) == sub_train:      
            fft_ref = fftpack.fft(train_ref[train_min:train_max])
            fft_test = fftpack.fft(train_test[train_min:train_max])
            
            # multiply in Fourier space
            spect = np.multiply(fft_ref,np.conj(fft_test))
            fft_corr.append(spect)
        i_tr += 1
        
    # average in Fourier space over sub-trains
    fft_corr_mean = np.mean(fft_corr, axis = 0) 
    
    # revert to real space
    corr = np.real(fftpack.ifft(fft_corr_mean))
    return corr

def correlogram(spikes_ref, spikes_test, state):
    corr_all = []
    for i_jit in range(n_jitter): # trial
        rate_ref = len(spikes_ref)/spikes_ref[-1]
        rate_test = len(spikes_test)/spikes_test[-1]

        # jitter spike times
        if n_jitter > 1:
            print (i_jit)
            spikes_ref = jitter(spikes_ref,stddev_jitter)
            spikes_test = jitter(spikes_test,stddev_jitter)
        
        # make spike trains
        train_ref = spike_train(spikes_ref, bin_dur)
        train_test = spike_train(spikes_test, bin_dur)
        
        # cut out state segment of interest
        train_ref = filter_spikes_by_periods(train_ref, state)
        train_test = filter_spikes_by_periods(train_test, state)
        
        # equalise lengths
        train_ref, train_test = \
            spike_train_length_equalize(train_test, train_ref)
            
        # compute correlogram
        corr = correlation_fft(train_ref, train_test)
        corr_all.append(corr)

    # mean over trials and normalisation
    corr_mean = np.dot(1/bin_dur/np.sqrt(rate_ref*rate_test), 
                           fftpack.fftshift(np.mean(corr_all, axis = 0)))
    # filtering out centre peak
#    corr_mean = np.multiply(corr_mean, filter_centre(half_peak_width))
    return np.real(corr_mean)

def expo_fit(xfit, paramA, paramB, paramC):
    yfit = paramA*np.exp(-1.*np.dot(xfit,paramB)) + paramC
    return yfit

def offset(ii,length):
    return length + ii*(length + 1) - ii*(ii-1)/2 - 1

#%% file reading deltaF/F with thresholding
folderpath = '/home/nghiem/Bureau/MaxEnt/Arjun_data/deltaff_062020/'
mouse_names_all = ['m658', # 
'm721',#1
'm724',#
'm724day1',#3
'm738',#4
'm785',#5
'm825',#
'm889127',#7
'm889127day1',#8x
'm889129',#9
'm889130']#10
#mousenum = 4

#rates for thresholding, in Hz? [2., 4., 6., 8., 10.]

rate_above_choose = 4.


for mousenum in [10]:#range(11):
    filename = 'deltaff_gray_062020_'+mouse_names_all[mousenum]+'.npy'
    
    data = np.load(folderpath + filename)
    mousename = '062020_' + mouse_names_all[mousenum]
    mousename += 'thresh'+str(rate_above_choose)
    name = mousename
    
    
    Nrec = 1
    if len(data.shape) == 4: # multiple sessions
        # concatenate all sessions together!
        Nrec = data.shape[0]
        data = np.concatenate(np.concatenate(data, axis = 0), axis = 1)
    else:
        data = np.concatenate(data)
    
    Nneu, Nbins = data.shape
    #    print(mouse_names_all[mousenum], Nneu)
    #%% DeltaF/F histograms by cell...
    plt.rcParams.update({'font.size': 15})
    #    plt.figure()
    #    for i_cell in range(Nneu):
    #        plt.hist(data[i_cell], alpha = 0.5)
    #    plt.xlabel('DeltaF/F')
    #    plt.ylabel('counts')
    
    #%%
    bin_dur = Nrec*5*60/Nbins # in seconds, as 5 minute recording epochs
    times = np.arange(Nbins)*bin_dur
    #    plt.figure()
    #    for i_cell in range(Nneu):
    #        plt.plot(times, data[i_cell]+2*i_cell, 'k')
    #    plt.xlabel('Time')
    #    
    
    
    #%%
    #    plt.figure()
    thresh_all = np.linspace(np.min(data), np.max(data), 1000)
    
    pA_thresh = []
    for thresh in thresh_all:
        data_thresh = np.zeros_like(data)
        data_thresh[data > thresh] = 1
        pA_thresh.append(np.mean(data_thresh))
    pA_thresh = np.array(pA_thresh)
    #    plt.semilogy(thresh_all, pA_thresh)
    #    plt.xlabel('Threshold')
    #    plt.ylabel('Mean activity rate (event/bin)')
    #    plt.title(mousename)
    
    #%% pick threshold
    thresh_choose = thresh_all[np.where(pA_thresh*Nbins/(5.*Nrec*60) \
                                        < rate_above_choose)[0][0]]
    
    data_binary = np.zeros_like(data)
    data_binary[data > thresh_choose] = 1
    
    Nneu_all, n_bins = data_binary.shape        
    #%% prepare data into dictionary
    Nneu_all, n_bins = data_binary.shape
    state_select = name
    state_title = state_select
    
    cells_all = {}
    for i_cell in range(Nneu_all):
        cells_all[i_cell] = {}
        cells_all[i_cell]['spike_train_'+state_select] = data_binary[i_cell]
    
    #%% exclude always silent cells
    cells_clean_all_nonzero = np.concatenate(np.argwhere(np.sum(data_binary, axis = 1) > 0))
    
    Nsubsamps = 1
    Dkl_all = []
    mousename_base = mousename
    for i_sub in range(Nsubsamps):
        #######TEST SUMBSAMPLE
        Nneu = 40#len(cells_clean) # subsample size
#        cells_clean = np.sort(np.random.choice(cells_clean_all_nonzero, 
#                                       size = Nneu, replace = False))
        mousename = mousename_base + 'subsampleN'+str(Nneu)+'_'+ str(i_sub) #+ 'allcells'#+
            #len(cells_clean) # subsample size
        cells_clean = np.load('cells_subsampled_'+mousename+'.npy')
        Nneu = len(cells_clean)
        print('Nneu = ', Nneu)
        #######END TEST SUBSAMPLE
        
        # generating vector of observables P
        P = []
        pA = [] # mean of P
        
        P_csr = [] # sparse copies
        
        
        
        #%%'''###############Inference parameters computation##########################'''
        
        
                       
        cells_clean_all = []
        for i_clean in cells_clean:
            P_csr.append(scipy.sparse.csr_matrix(cells_all[i_clean]\
                ['spike_train_' + state_select]))
            pA.append(P_csr[-1].mean())
            cells_clean_all.append(i_clean)
        
        for i_clean0 in cells_clean:
            for i_clean1 in cells_clean:
                if i_clean0 < i_clean1:#np.in1d([i_clean0, i_clean1], cells_exclude).any() == False \
                    #and i_clean0 < i_clean1:
                    max_len = np.amin([len(cells_all[i_clean0]['spike_train_'+ state_select]), 
                                     len(cells_all[i_clean1]['spike_train_'+ state_select])])
                    train0 = scipy.sparse.csr_matrix(cells_all[i_clean0]\
                        ['spike_train_' + state_select ][:max_len])
                    train1 = scipy.sparse.csr_matrix(cells_all[i_clean1]\
                        ['spike_train_'+ state_select][:max_len])
                    P_csr.append(train0.multiply(train1))
                    pA.append(P_csr[-1].mean())

                    
        # generating susceptibility matrix chiA (Fisher matrix)
        n_cells_clean = len(cells_clean_all)            
        chiA = np.zeros((len(P_csr), len(P_csr)))
        chiAC = np.zeros((len(P_csr), len(P_csr))) # connected correlations
        
        for i_chiA0 in range(len(P_csr)):
            for i_chiA1 in range(i_chiA0, len(P_csr)):
                max_len = np.amin([P_csr[i_chiA0].shape[1], P_csr[i_chiA1].shape[1]])
                arr0 = P_csr[i_chiA0][0, :max_len]
                arr1 = P_csr[i_chiA1][0, :max_len]
                chiA[i_chiA0][i_chiA1] = arr0.multiply(arr1).mean()
                chiAC[i_chiA0][i_chiA1] = chiA[i_chiA0][i_chiA1]
                # connecting 2-way correlations
        #        if i_chiA0<n_cells_clean and i_chiA1<n_cells_clean:
                chiAC[i_chiA0][i_chiA1] -= pA[i_chiA0]*pA[i_chiA1]
                chiAC[i_chiA1][i_chiA0] = chiAC[i_chiA0][i_chiA1]
                chiA[i_chiA1][i_chiA0] = chiA[i_chiA0][i_chiA1]
        np.save('chiA_'+state_title, chiA)
        np.save('chiAC_'+state_title, chiAC)
        #
        #chiA = np.load('chiA_'+state_title+'.npy')
        #chiAC = np.load('chiAC_'+state_title+'.npy')
                
                
        #%%'''############################Inference####################################'''
        eta = 5e-3#0.5e-3# for regularisation; control: 1e-4
        logTd = 5
        nStepMore = 10# thermalisation steps
        nWork = 1#multiprocessing.cpu_count() # number of cores for parallelisation
        #n_cells_clean = n_cells - len(cells_exclude)
        n_bins = len(cells_all[i_cell]['spike_train_' + state_select])
        
        eVS = np.sort(LA.eigh(chiA)[0]) # eigenvalues of the Fisher Matrix
        labels = np.load('labels_2clusters_kmeans_residuals_'+mousename+'.npy')[:Nneu]
        #placeholder, just fill with any vector of Nneu integers if not using
        q, jList, pK_all, pK_types, pK_indep, Kt_types, raster_MC = optimalLearningParallel(\
                                                                    n_cells_clean, 
                                                                    n_bins, 
                                                                    np.min((2/eVS[0], 
                                                                           n_bins)), 
                                                                    logTd, pA, chiAC, 
                                                                    eta, 1.0, 
                                                                    nStepMore, nWork, 
                                                                    labels)
        np.save('jList_'+mousename+'.npy', jList)
        np.save('pK'+mousename+'.npy', pK_all)
        np.save('raster_MC_'+mousename+'.npy', raster_MC)
        #%%'''######################Inference results analysis#########################'''
        
        # connected and pearson correlations
        c_data = []
        c_MC = []
        corr_pearson_data = []
        corr_pearson_norm_data = []
        corr_pearson_MC = []
        for ic in range(n_cells_clean):
            for jc in range(ic, n_cells_clean):
                # connected correlation
                c_data.append(pA[MCspins.offset(ic,n_cells_clean)+jc] - pA[ic]*pA[jc])
                c_MC.append(q[MCspins.offset(ic,n_cells_clean)+jc] - q[ic]*q[jc])
                # Pearson correlation
                denom_sq_data = chiA[ic][ic]*chiA[jc][jc]
                corr_pearson_data.append(c_data[-1]/np.sqrt(denom_sq_data))
                corr_pearson_norm_data.append(c_data[-1]/np.sqrt(denom_sq_data)\
                    /np.sqrt(pA[ic]*pA[jc]))
                denom_sq_MC = (q[ic]-q[ic]**2)*(q[jc]-q[jc]**2)
                corr_pearson_MC.append(c_MC[-1]/np.sqrt(denom_sq_MC))
        
        plt.plot(c_data, c_MC, '.')
        plt.plot(c_data,c_data,'k')
        plt.xlabel('Pairwise correlations -  DATA')
        plt.ylabel('Pairwise correlations -  MODEL')
        
        print('Pearson pairwise cov = ', stats.pearsonr(c_data, c_MC))
        
        #%% P(K) whole population
        plt.figure()
        pK_data = np.zeros(n_cells_clean+1)
        pK_MC = pK_all[-1]/np.sum(pK_all[-1])
        for i_bin in range(n_bins):
            magn = 0
            for i_cell in np.array(cells_clean_all):
                if len(cells_all[i_cell]['spike_train_' + state_select]) > i_bin:
                    magn += int(cells_all[i_cell]['spike_train_'+ state_select][i_bin])
            pK_data[magn] += 1.0/n_bins
        err_MC = np.sqrt((pK_MC - np.multiply(pK_MC, pK_MC))/n_bins)
        err_data = np.sqrt((pK_data - np.multiply(pK_data, pK_data))/n_bins)
        
        #%% plotting P(K)
        plt.rcParams.update({'font.size': 15})
        
        
        import plot_figs as pfigs
        pfigs.plot_pK_inset([pK_data, pK_MC, pK_indep], [n_bins, n_bins, n_bins], 
                      ['data', 'pairwise model', 'independent model'], 
                          pK_indep = None, state_title = name)
        np.save('pK_MC_'+state_title+'.npy', pK_MC)
        #%% KL divergence
        pK_MC = pK_all[-1]/np.sum(pK_all[-1])
    #    if np.any(pK_data == 0) and np.any(pK_MC == 0):
    #        idx_max = np.min((np.where(pK_MC == 0)[0][0],
    #                          np.where(pK_data == 0)[0][0]))
    #    else:
    #        idx_max = np.min((len(pK_data), len(pK_MC)))
            
        idx_nonzero = np.where(pK_MC*pK_data > 0)[0]
        Dkl = -np.sum(pK_MC[idx_nonzero]*np.log(pK_data[idx_nonzero]/pK_MC[idx_nonzero]))
        print(mousename + 'KL Divergence = ', Dkl)
        Dkl_all.append(Dkl)
    
    np.save('Dkl_ising_allmice_thresh'+str(rate_above_choose)+mousename+'subsamp20.npy',
            Dkl_all)
    #%% p(K) by type
    
    #Ntypes = np.max(labels) + 1
    #plt.figure()
    #plt.yscale('log')
    #col_cycle = ['b','r']
    #for i_type in range(Ntypes): 
    #    pK_data = np.zeros(n_cells_clean+1)
    #    pK_MC = pK_types[i_type][0]/np.sum(pK_types[i_type][0])
    #    for i_bin in range(n_bins):
    #        magn = 0
    #        for i_cell in np.array(cells_clean_all)[labels == i_type]:
    #            if len(cells_all[i_cell]['spike_train_' + state_select]) > i_bin:
    #                magn += cells_all[i_cell]['spike_train_'+ state_select][i_bin]
    #        pK_data[magn] += 1.0/n_bins
    #    err_MC = np.sqrt((pK_MC - np.multiply(pK_MC, pK_MC))/n_bins)
    #    err_data = np.sqrt((pK_data - np.multiply(pK_data, pK_data))/n_bins)
    #    Kmax_MC = np.where(pK_MC == 0)[0][0]
    #    Kmax_data = np.where(pK_data == 0)[0][0]
    #    
    ##    plt.figure(figsize = figsize_rect)
    #    plt.errorbar(range(Kmax_MC), pK_MC[:Kmax_MC], yerr = err_MC[:Kmax_MC], 
    #                 color = col_cycle[i_type], linestyle = '--',
    #                 ecolor = col_cycle[i_type], label = 'Ising', linewidth = 2.0)
    #    plt.errorbar(range(Kmax_data),pK_data[:Kmax_data], yerr = err_data[:Kmax_data],
    #                 color = col_cycle[i_type], 
    #                 ecolor = col_cycle[i_type], label = 'Data', linewidth = 2.0)
    ##    plt.semilogy(range(25), pK_indep[:25],'k',  
    ##                 label = 'Independent',linewidth=2.0)
    #plt.xlabel("K")
    #plt.ylabel("p(K)")    
    #plt.legend()
    #plt.title("Population activity as given by data and model \nduring "+ \
    #    state_select + " states\n")
    #plt.rcParams.update({'font.size': 15})      
    #plt.savefig('pK_bytype_'+state_title+'.pdf')
    #
    #
    #
    #%% p(K,K) for two cell types or clusters only
    
    #Ntypes = 2
    #plt.figure()
    #Kmax_all = np.zeros(Ntypes)
    #K_all = []
    #for i_type in range(Ntypes): 
    #    Kmax_all[i_type] = np.max(Kt_types[i_type]) + 1
    #    K_all.append(np.arange(Kmax_all[i_type]).astype(int))
    #
    #kcombinations = list(itertools.product(*K_all)) 
    #
    #pKK_MC = np.zeros(list(Kmax_all.astype(int)))
    #for comb in kcombinations:
    #    idx_kk = np.zeros((Ntypes,n_bins))
    #    for i_type in range(Ntypes):
    #        idx_kk[i_type] = Kt_types[i_type] == comb[i_type]
    #    
    #    pKK_MC[comb] = np.mean(np.prod(idx_kk, axis = 0))
    #
    #pK_types[0] /= np.sum(pK_types[0])
    #pK_types[1] /= np.sum(pK_types[1])
    #plt.imshow(np.log(pKK_MC/np.outer(pK_types[0][0,:int(Kmax_all[0])], 
    #                                  pK_types[1][0,:int(Kmax_all[1])])), 
    #    cmap = 'RdBu_r', interpolation = 'nearest', 
    #           vmin = -5, vmax = 5)
    #plt.colorbar()
    #plt.savefig('pKK_ising'+state_title+'.pdf')
    

# %%
