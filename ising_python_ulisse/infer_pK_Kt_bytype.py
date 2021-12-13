#!/usr/bin/env python
#coding=utf-8

'''Infers Ising model parameters from data, using Data-Driven Max-Ent method

Algorithm description (and consistent notations):
Ferrari U., Learning maximum entropy models from finite-size data sets: A fast
data-driven algorithm allows sampling from the posterior distribution, 
Phys. Rev. E 94, 023301 – 1 August 2016

Translated from Ulisse's Matlab code.

22/09/2016 T.A. Nghiem trang-anh.nghiem@cantab.net
'''
#%%
import numpy as np
import ctypes
from ctypes import util
import os 
#from joblib import Parallel, delayed
#import multiprocessing

#os.add_dll_directory("C:/Users/auror/New_VAES/VAEs/ising_python_ulisse")
os.chdir("C:/Users/auror/New_VAES/VAEs/ising_python_ulisse")
print(os.getcwd())
MCspins = ctypes.WinDLL('./MCspins.dll') 
MCspinsPK = ctypes.CDLL('./MCspinsPK.so')
MCspinsPKbytype = ctypes.CDLL('./MCspinsPKbytype.so')
MCspinsPKbytypeRaster = ctypes.CDLL('./MCspinsPKbytypeRaster.so')

#MCspins = ctypes.CDLL('/Users/auror/New_VAES/VAEs/ising_python_ulisse/MCspins.so') 
#MCspinsPK = ctypes.CDLL('/Users/auror/New_VAES/VAEs/ising_python_ulisse/MCspinsPK.so')
#MCspinsPKbytype = ctypes.CDLL('/Users/auror/New_VAE S/VAEs/ising_python_ulisse/MCspinsPKbytype.so')
#MCspinsPKbytypeRaster = ctypes.CDLL('/Users/auror/New_VAES/VAEs/ising_python_ulisse/MCspinsPKbytypeRaster.so')
#MCspins.pyMC.restype = ctypes.POINTER(ctypes.c_double)
#%%
def epsilon(gradient, B, D, chiEta):
    return np.sqrt(0.5*B/D*(np.dot(gradient,(np.dot(np.linalg.inv(chiEta), 
                                                gradient)))))

def delta_jList(gradient, alpha, chiEta):
    return alpha * np.dot(np.linalg.inv(chiEta), gradient)

def cpyMC(inputs):
    N, nWork, pMatTry, lattice, nW, Beff, logTd, jList = inputs
    # converting to C doubles and ints
    jList_c = (ctypes.c_double * len(jList))(*jList)
    pMatTry_c = (ctypes.c_double * len(pMatTry[nW]))(*pMatTry[nW])        
    lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
    lattice_out_c = lattice_c
    # run Monte-Carlo
    MCspins.pyMC(N,
        int(Beff/nWork),
        int(np.floor(np.log2(Beff)-4)),
        int(logTd),
        ctypes.byref(jList_c),
        ctypes.byref(lattice_c),
        ctypes.byref(pMatTry_c),
        ctypes.byref(lattice_out_c))
    # copying back to python arrays
    for ip in range(len(pMatTry[nW])): 
        pMatTry[nW][ip] = pMatTry_c[ip]
    for il in range(len(lattice[nW])): 
        lattice[nW][il] = lattice_out_c[il]
    return
                            

def run_longMC(jList, Nsim, N, B, Bmin, logTd, p, name = 'presyn_738'):
    D = int(N*(N + 1)/2) 
    
    pMat= np.zeros((Nsim, D))
    pK = np.zeros((Nsim, N+1)) # probability of population firing 
    # converting to C doubles
    jList_c = (ctypes.c_double * len(jList))(*jList)
  
    # Initialisation lattice
    lattice = np.random.rand(Nsim, N) < \
        np.tile(p[:N],Nsim).reshape(Nsim,len(p[:N])).astype(np.uint32)
    
    
    # run Nsim full simulations
    for nW in range(Nsim):
        # coverting from python to C objects
        pMat_c = (ctypes.c_double * len(pMat[nW]))(*pMat[nW])        
        pK_c = (ctypes.c_double * len(pK[nW]))(*pK[nW])        
        lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
        lattice_out_c = lattice_c
        
        MCspinsPK.pyMC(N,
            int(B),
            int(np.floor(np.log2(B)-4)),
            int(logTd),
            ctypes.byref(jList_c),
            ctypes.byref(lattice_c),
            ctypes.byref(pMat_c),
            ctypes.byref(pK_c),
            ctypes.byref(lattice_out_c))
        
            # copying back to python arrays
        for ip in range(len(pMat[nW])): 
            pMat[nW][ip] = pMat_c[ip]
        for il in range(len(lattice[nW])): 
            lattice[nW][il] = lattice_out_c[il]
        for ip in range(len(pK[nW])): 
            pK[nW][ip] = pK_c[ip]
        
        print(nW, pK[nW])
    
    np.save('pK_longsim'+name+'.npy', pK)
    return pK

                    
def optimalLearningParallel(N, B, Bmin, logTd, p, chi, eta, 
                            epsThreshold, nStepMore, nWork, labels):
    # Parameters
    D = int(N*(N + 1)/2) 
    alphaMin = 10**(-7)
    alphaMax = 2.0
    alpha = 1.0

    # Initialisation lattice
    lattice = np.random.rand(nWork, N) < \
        np.tile(p[:N],nWork).reshape(nWork,len(p[:N])).astype(np.uint32)
    q = np.zeros(len(p))
    
    # Indep model
    JtOpMc = np.zeros(int(N*(N-1)/2))
    HtOpMc = np.log(np.multiply(p[:N], 1/np.subtract(1, p[:N] )))
    jList = np.concatenate((HtOpMc, JtOpMc))
        
    Btest = B
    for n1 in range(N):
        q[n1] = p[n1]    
        for n2 in range((n1+1),N):
            print('n1',str(n1), 'n2', str(n2), 'offset', MCspins.offset(n1,N)+n2)
            q[ MCspins.offset(n1,N)+n2 ] = p[n1] * p[n2]
            # independent model: proba of n1 and n2 is the product of probas
    # Monte-carlo for indep model
    pMat= np.zeros((nWork, D))
    pK = np.zeros((nWork, N+1)) # probability of population firing 
    # converting to C doubles
    jList_c = (ctypes.c_double * len(jList))(*jList)
    for nW in range(nWork):
        pMat_c = (ctypes.c_double * len(pMat[nW]))(*pMat[nW])        
        pK_c = (ctypes.c_double * len(pK[nW]))(*pK[nW])        
        lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
        lattice_out_c = lattice_c
        
        MCspinsPK.pyMC(N,
            int(Btest/nWork),
            int(np.floor(np.log2(B)-4)),
            int(logTd),
            ctypes.byref(jList_c),
            ctypes.byref(lattice_c),
            ctypes.byref(pMat_c),
            ctypes.byref(pK_c),
            ctypes.byref(lattice_out_c))
            # copying back to python arrays
        for ip in range(len(pMat[nW])): 
            pMat[nW][ip] = pMat_c[ip]
        for il in range(len(lattice[nW])): 
            lattice[nW][il] = lattice_out_c[il]
        for ip in range(len(pK[nW])): 
            pK[nW][ip] = pK_c[ip]
    q = np.mean(pMat, axis = 0)
    pK_indep = np.mean(pK, axis = 0)
    print (pK_indep)
            
    chiEta = chi + np.diag(np.ones(len(chi))*eta)#eta
    
#    fOfX = 0#np.random.normal(-np.dot(eta,jList), eta/B ) # for regularisation
    grad = np.subtract(p, q)  
    print(grad)
    epsOpMc = epsilon(grad, B, D, chiEta)
    epsOpMcOld = epsOpMc
    
    step = 0
    Beff = np.max([np.min([B/(epsOpMcOld**2),B]),Bmin])
    
#    output = [step, alpha , epsOpMc , Beff]
    print(np.log(epsOpMc))
    
    while epsOpMc > epsThreshold: # prediction-data error still large
        gradOpMc =np.subtract(p, q)
        jList = np.add(jList,delta_jList(gradOpMc, alpha, chiEta))
        pMatTry = np.zeros((nWork,D))
        
        
        jList_c = (ctypes.c_double * len(jList))(*jList)
        
#        inputs = [None]*nWork
#        for nW in range(nWork):  
#            inputs[nW] = (N, nWork, pMatTry, lattice, nW, Beff, logTd, jList)
#        Parallel(n_jobs=nWork)(delayed(cpyMC)(i) for i in inputs)
        for nW in range(nWork): # run Monte-Carlo simulation
            # converting to C doubles and ints
            pMatTry_c = (ctypes.c_double * len(pMatTry[nW]))(*pMatTry[nW])        
            lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
            lattice_out_c = lattice_c
            # run Monte-Carlo
            MCspins.pyMC(N,
                int(Beff/nWork),
                int(np.floor(np.log2(Beff)-4)),
                int(logTd),
                ctypes.byref(jList_c),
                ctypes.byref(lattice_c),
                ctypes.byref(pMatTry_c),
                ctypes.byref(lattice_out_c))
            # copying back to python arrays
            for ip in range(len(pMatTry[nW])): 
                pMatTry[nW][ip] = pMatTry_c[ip]
            for il in range(len(lattice[nW])): 
                lattice[nW][il] = lattice_out_c[il]
        pTry = np.mean(pMatTry, axis = 0)
        step += 1
        
#        fOfX = np.random.norm(-(eta*jList), eta/B ) 
        gradTry = np.subtract(p, pTry)
        epsOpMc = epsilon(gradTry, B, D, chiEta)
        dEps = epsOpMc - epsOpMcOld
        
        if dEps<0: # better estimate, update model q 
            alpha = np.min([alphaMax,alpha*1.05])
            q = pTry
            epsOpMcOld = epsOpMc
        else: # worse estimate, reduce step size
            jList = np.subtract(jList, delta_jList(gradOpMc, alpha, chiEta))
            alpha = np.max([alpha/np.sqrt(2.0),alphaMin])     
            
            # converting to C doubles
            jList_c = (ctypes.c_double * len(jList))(*jList)
            
            pMatTry = np.zeros((nWork,D))
            for nW in range(nWork): 
                # converting to C doubles and ints
                pMatTry_c = (ctypes.c_double * len(pMatTry[nW]))(*pMatTry[nW])        
                lattice_c = (ctypes.c_uint32 *len(lattice[nW]))(*lattice[nW])
                lattice_out_c = lattice_c
                # run Monte-Carlo
                MCspins.pyMC(N,
                    int(Beff/nWork),
                    int(np.floor(np.log2(Beff)-4)),
                    int(logTd),
                    ctypes.byref(jList_c),
                    ctypes.byref(lattice_c),
                    ctypes.byref(pMatTry_c),
                    ctypes.byref(lattice_out_c))
                
                # copying back to python arrays
                for ip in range(len(pMatTry[nW])): 
                    pMatTry[nW][ip] = pMatTry_c[ip]
                for il in range(len(lattice[nW])): 
                    lattice[nW][il] = lattice_out_c[il]
#            fOfX = np.random.norm(-(eta*jList), eta/B ) 
            grad = np.subtract(p, q)
            epsOpMcOld = epsilon(grad, B, D, chiEta)
    
            step += 1
        
        if step%1==0:
            print(step , alpha, Beff/B, np.log(epsOpMc))
    
        Beff = np.max([np.min([B/(epsOpMcOld**2),B]),Bmin])
#        output.append([step, alpha , epsOpMc , Beff])
    
    print('Inference done, now thermalization')
    
    logTd += 6    
    
    gradOpMc = np.subtract(p, q) 
    pK_all = []
    for ss in range(nStepMore):    
        jList = np.add(jList, delta_jList(gradOpMc, alpha, chiEta))
        
        pMat= np.zeros((nWork, D))
        pK = np.zeros((nWork, N+1)) # probability of population firing 
        # converting to C doubles
        jList_c = (ctypes.c_double * len(jList))(*jList)
    
        
#        inputs = []*nWork
#        for nW in range(nWork):           
#            inputs[nW] = (N, nWork, pMat, lattice, nW, Beff, logTd, jList)
#        Parallel(n_jobs=nWork)(delayed(cpyMC)(i) for i in inputs)        
        for nW in range(nWork):
            pMat_c = (ctypes.c_double * len(pMat[nW]))(*pMat[nW])        
            pK_c = (ctypes.c_double * len(pK[nW]))(*pK[nW])        
            lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
            lattice_out_c = lattice_c
            
            MCspinsPK.pyMC(N,
                int(Beff/nWork),
                int(np.floor(np.log2(Beff)-4)),
                int(logTd),
                ctypes.byref(jList_c),
                ctypes.byref(lattice_c),
                ctypes.byref(pMat_c),
                ctypes.byref(pK_c),
                ctypes.byref(lattice_out_c))
            
                # copying back to python arrays
            for ip in range(len(pMat[nW])): 
                pMat[nW][ip] = pMat_c[ip]
            for il in range(len(lattice[nW])): 
                lattice[nW][il] = lattice_out_c[il]
            for ip in range(len(pK[nW])): 
                pK[nW][ip] = pK_c[ip]
        q = np.mean(pMat, axis = 0)
#        chi_out = np.cov(pMat)
        pK_all.append(np.mean(pK, axis = 0))
        
#        fOfX = np.random.norm(-(eta*jList), eta/B ) 
        gradOpMc = np.subtract(p, q)  
        epsOpMc =  epsilon(gradOpMc, B, D, chiEta)
        alpha = alpha/epsOpMc
        print(step, alpha, ss, np.log(epsOpMc), pK_all[-1][2])
    
    np.save('jList_spontaneous_control.npy', jList)
    # returning K_t for each population and separate p(K) only for last step...
    Ntypes = np.max(labels) + 1
    pK_types = []
    Kt_types = []
    for type_curr in range(Ntypes):
        ctypes._reset_cache()
        # save full raster
        lattice_out = np.zeros(N*int(Beff)).astype(int)
        
        idx_types = np.arange(N)[labels == type_curr]
        lentype = len(idx_types)
        
        K_t = np.zeros(B).astype(int) # population activity in time K(t)
        for nW in range(nWork):
            pMat_c = (ctypes.c_double * len(pMat[nW]))(*pMat[nW])        
            pK_c = (ctypes.c_double * len(pK[nW]))(*pK[nW])        
            lattice_c = (ctypes.c_uint32 * len(lattice[nW]))(*lattice[nW])
            lattice_out_c = (ctypes.c_uint32 * len(lattice_out))(*lattice_out)
#            lattice_out_c = lattice_c
            K_t_c = (ctypes.c_uint32 * len(K_t))(*K_t)
            idx_types_c = (ctypes.c_uint32 * len(idx_types))(*idx_types)
            
            MCspinsPKbytypeRaster.pyMC(N,
                int(lentype),
                int(Beff/nWork),
                int(np.floor(np.log2(Beff)-4)),
                int(logTd),
                ctypes.byref(jList_c),
                ctypes.byref(idx_types_c),
                ctypes.byref(lattice_c),
                ctypes.byref(pMat_c),
                ctypes.byref(pK_c),
                ctypes.byref(lattice_out_c), 
                ctypes.byref(K_t_c))
            
                # copying back to python arrays
            for ip in range(len(pMat[nW])): 
                pMat[nW][ip] = pMat_c[ip]
            for il in range(len(lattice[nW])): 
                lattice[nW][il] = lattice_out_c[il]
            for ilo in range(len(lattice_out)): 
                lattice_out[ilo] = lattice_out_c[ilo]
            for ip in range(len(pK[nW])): 
                pK[nW][ip] = pK_c[ip]
            for ip in range(len(K_t)): 
                K_t[ip] = K_t_c[ip]
#        q = np.mean(pMat, axis = 0)
    #        chi_out = np.cov(pMat)
        pK_types.append(pK)    
        Kt_types.append(K_t)    
        print(np.sum(lattice_out))
#        np.save('lattice_ising_spontaneous_control35.npy',lattice_out)
#        if ss%1==0:
    return q, jList, pK_all, pK_types, pK_indep, Kt_types, lattice_out
