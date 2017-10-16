#!/usr/bin/env python

from __future__ import print_function

import emcee
from emcee.utils import MPIPool
import numpy as np

#import scipy.optimize as op
#import matplotlib.pyplot as pl
#from matplotlib.ticker import MaxNLocator
#import sys

import Cmodules_combined_win_local
import util as UT 

def match_para(theta, free_para, fix_para):
    value_array = np.arange(len(free_para),dtype=np.float)

    counter = 0
    if(free_para[0] == True):
        value_array[0] = theta[counter]
        counter += 1
    else: value_array[0] = fix_para[0]
    if(free_para[1] == True):
        value_array[1] = theta[counter]
        counter += 1
    else: value_array[1] = fix_para[1]
    if(free_para[2] == True):
        value_array[2] = theta[counter]
        counter += 1
    else: value_array[2] = fix_para[2]
    if(free_para[3] == True):
        value_array[3] = theta[counter]
        counter += 1
    else: value_array[3] = fix_para[3]
    if(free_para[4] == True):
        value_array[4] = theta[counter]
        counter += 1
    else: value_array[4] = fix_para[4]
    if(free_para[5] == True):
        value_array[5] = theta[counter]
        counter += 1
    else: value_array[5] = fix_para[5]
    if(free_para[6] == True):
        value_array[6] = theta[counter]
        counter += 1
    else: value_array[6] = fix_para[6]
    if(free_para[7] == True):
        value_array[7] = theta[counter]
        counter += 1
    else: value_array[7] = fix_para[7]
    if(free_para[8] == True):
        value_array[8] = theta[counter]
        counter += 1
    else: value_array[8] = fix_para[8]
    if(free_para[9] == True):
        value_array[9] = theta[counter]
        counter += 1
    else: value_array[9] = fix_para[9]
    if(free_para[10] == True):
        value_array[10] = theta[counter]
        counter += 1
    else: value_array[10] = fix_para[10]

    return value_array


def lnprior(theta, free_para, fix_para):
    value_array = match_para(theta, free_para, fix_para)
    
    if 0.5 < value_array[0] < 1.5 and\
       0.5 < value_array[1] < 1.5 and\
       0.1 < value_array[2] < 1.1 and\
       0.3 < value_array[3] < 5. and\
       0.3 < value_array[4] < 5. and\
       -6. < value_array[5] < 6. and\
       -6. < value_array[6] < 6. and\
       -10000 < value_array[7] < 10000 and\
       -10000 < value_array[8] < 10000 and\
       0.5 < value_array[9] < 15. and\
       0.5 < value_array[10] < 15.:
        return 0.
    return -np.inf


def lnlike(theta, x, y, y_SGC, binrange1, binrange2, binrange3, maxbin1, Cinv, Cinv_SGC, free_para, fix_para):
    value_array = match_para(theta, free_para, fix_para)
    
    modelX = np.arange(2*(binrange1+binrange2+binrange3),dtype=np.float)
    
    if value_array[0] > 0.8 and value_array[0] < 1.4 and value_array[1] > 0.8 and value_array[1] < 1.4:
        modelX = Cmodules_combined_win_local.taruya_model_module_combined_win_local(100, binrange1, binrange2, binrange3, maxbin1, x, value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], value_array[6], value_array[7], value_array[8], value_array[9], value_array[10])
    
        dummy_model_NGC = modelX[0]
        #print("dummy_model_NGC = ", dummy_model_NGC, len(x))
        dummy_model_SGC = modelX[1]

        #for i in range(0, 37):
        #    print("k = ", x[i], dummy_model_NGC[i], dummy_model_SGC[i])

        '''
        pl.clf()

        pl.plot(x, dummy_model_NGC, "k", lw=2, label='model')
        pl.plot(x, dummy_model_SGC, "b", lw=2, label='model')
        pl.errorbar(x, y, yerr=1, fmt=".k")
        pl.errorbar(x, y_SGC, yerr=1, fmt=".k")
        pl.legend(loc=1)
        pl.ylim(100, 200000)
        pl.xlabel("$k [h/Mpc]$")
        pl.ylabel("$P(k) [Mpc/h]^{3}$")
        pl.yscale("log")
        #pl.savefig("runA_output/test.png")
        #pl.show()
        '''
        
        diff = dummy_model_NGC - y
        diff_SGC = dummy_model_SGC - y_SGC
        chi2 = np.dot(diff,np.dot(Cinv,diff))
        chi2_SGC = np.dot(diff_SGC,np.dot(Cinv_SGC,diff_SGC))

        print("alpha_perp: ", value_array[0],"alpha_para: ", value_array[1], "fsig8: ", value_array[2], "b1NGCsig8: ", value_array[3], "b1SGCsig8: ", value_array[4], "b2NGCsig8: ", value_array[5], "b2SGCsig8: ", value_array[6], "NNGC: ", value_array[7], "NSGC: ", value_array[8], "sigmaNGC_v: ", value_array[9], "sigmaSGC_v: ", value_array[10], ", value is: ", chi2+chi2_SGC, " chi2_SGC = ", chi2_SGC, " chi2 = ", chi2)

        return -0.5*(chi2 + chi2_SGC)
            
    return -0.5*(10000.)

# Define the probability function as likelihood * prior.
def lnprob(theta, x, y, y_SGC, binrange1, binrange2, binrange3, maxbin1, Cinv, Cinv_SGC, free_para, fix_para):
    lp = lnprior(theta, free_para, fix_para)
    
    value_array = match_para(theta, free_para, fix_para)
    
    if not np.isfinite(lp):
        dummy = -np.inf
    dummy = lp + lnlike(theta, x, y, y_SGC, binrange1, binrange2, binrange3, maxbin1, Cinv, Cinv_SGC, free_para, fix_para)

    return dummy


def gelman_rubin_convergence(withinchainvar, meanchain, n, Nchains, ndim):
    # Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter

    meanall = np.mean(meanchain, axis=0)
    W = np.mean(withinchainvar, axis=0)
    B = np.arange(ndim,dtype=np.float)
    for jj in range(0, ndim):
        B[jj] = 0.
    for jj in range(0, Nchains):
        B = B + n*(meanall - meanchain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./n)*W + B/n
    scalereduction = np.sqrt(estvar/W)

    return scalereduction

###########################################
###  Main program  ########################
###########################################

def main():
    rank = 0
    size = 1
    
    kbins = 60
    binsize = 120/kbins
    minbin1 = 2
    minbin2 = 2
    minbin3 = 2
    maxbin1 = 30
    maxbin2 = 30
    maxbin3 = 20
    binrange1 = maxbin1-minbin1
    binrange2 = maxbin2-minbin2
    binrange3 = maxbin3-minbin3
    totbinrange = binrange1+binrange2+binrange3
    tag = "nohex_combined_z1_binsize2_data_test"
    outpath = UT.dat_dir() 
    tag2 = "z1"
    ihex = 0
    imaxlike = 0
    Nchains = 4
    ithin = 1
    minlength = 600 #600
    ichaincheck = 100
    nwalkers = 48
    temperature = 2.e-3
    epsilon = 0.02 #0.02

    value_array = [1.00830111426, 1.0007368972, 0.478098423689, 1.33908539185, 1.33663505549, 1.15627984704, 0.31657562682, -1580.01689181, -928.488535962, 6.14815801563, 6.79551199595] #z1 max likelihood
    modelX = Cmodules_combined_win_local.taruya_model_module_combined_win_local(100, binrange1, binrange2, binrange3, maxbin1, 
            np.arange(0.01, 0.15, 0.005), value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], value_array[6], value_array[7], value_array[8], value_array[9], value_array[10])

    print(modelX[0])
    print(modelX[1])

    if(tag2 == "z3"):
        inputfile = "/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_NGC_z3_COMPnbar_TIC_340_650_360_120.dat"
        inputfile_SGC = "/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_SGC_z3_COMPnbar_TIC_190_500_280_120.dat"
    elif(tag2 == "z2"):
        inputfile = "/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_NGC_z2_COMPnbar_TIC_290_540_300_120.dat"
        inputfile_SGC = "/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_SGC_z2_COMPnbar_TIC_160_430_240_120.dat"
    elif(tag2 == "z1"):
        inputfile = "/Users/chang/projects/nonGaussLike/dat/Beutler/public_material_RSD/test.dat"
        #/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_NGC_z1_COMPnbar_TIC_250_460_260_120.dat"
        inputfile_SGC = "/Users/fbeutler/clustools/output/nbar_combined_output/data/ps1D_DR12_data_SGC_z1_COMPnbar_TIC_190_360_210_120.dat"
    
    if tag2 == "z1":
        NGCmocks = 2045
        SGCmocks = 2048
    elif tag2 == "z2":
        NGCmocks = 2045
        SGCmocks = 2048
    elif tag2 == "z3":
        NGCmocks = 2045
        SGCmocks = 2048
    
    npara = 11
    
    all_name = r"$\alpha_{perp}$", r"$\alpha_{para}$", r"$f\sigma_{8}$", r"$b^NGC_{1}\sigma_{8}$", r"$b^SGC_{1}\sigma_{8}$", r"$b^NGC_{2}\sigma_{8}$", r"$b^SGC_{2}\sigma_{8}$", "$N^NGC$", "$N^SGC$", r"$\sigma^NGC_{v}$", r"$\sigma^SGC_{v}$"
    
    # should the parameter be varied freely?
    free_para = True, True, True, True, True, True, True, True, True, True, True
    # if free_para is false use the value in fix_para
    fix_para = 1., 1., 0.43, 0., 0., 0., 0., 0., 0., 0., 0.
    
    # has the size ndim
    start = 1.00830111426, 1.0007368972, 0.478098423689, 1.33908539185, 1.33663505549, 1.15627984704, 0.31657562682, -1580.01689181, -928.488535962, 6.14815801563, 6.79551199595 #z1 max likelihood

#################################
    
    num_lines = sum(1 for line in open(inputfile))-31
    #num_lines = sum(1 for line in open(inputfile))-4
    if rank == 0: print("num_lines = ", num_lines, " binrange1 = ", binrange1)
    
    x1 = np.arange(totbinrange,dtype=np.float)
    y1 = np.arange(totbinrange,dtype=np.float)
    Nmodes = np.arange(totbinrange,dtype=np.float)
    x = np.arange(totbinrange/binsize,dtype=np.float)
    y = np.arange(totbinrange/binsize,dtype=np.float)
    
    # read in fitting data
    n22 = 0
    n33 = 0
    norm = 0.
    with open(inputfile,"r") as f:
        n = 0
        for i in range(0, 31):
            next(f)
        for line in f:
            dummy2 = map(float, line.split())
            if n >= minbin1 and n < maxbin1:
                Nmodes[n-minbin1] = dummy2[6]
                x1[n-minbin1] = dummy2[0]
                y1[n-minbin1] = dummy2[2]
                x1[binrange1+n-minbin1] = dummy2[0]
                y1[binrange1+n-minbin1] = dummy2[3]
                if n < maxbin3:
                    x1[binrange1+binrange2+n-minbin1] = dummy2[0]
                    y1[binrange1+binrange2+n-minbin1] = dummy2[4]
                n22 = n22 + 1
                norm = norm + Nmodes[n-minbin1]
                if(n22 >= binsize):
                    x[n33] = 0
                    y[n33] = 0
                    x[binrange1/binsize+n33] = 0
                    y[binrange1/binsize+n33] = 0
                    if n < maxbin3:
                        x[binrange1/binsize+binrange2/binsize+n33] = 0
                        y[binrange1/binsize+binrange2/binsize+n33] = 0
                    for j in range(0, binsize):
                        x[n33] = x[n33] + x1[n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        y[n33] = y[n33] + y1[n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        x[binrange1/binsize+n33] = x[binrange1/binsize+n33] + x1[binrange1+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        y[binrange1/binsize+n33] = y[binrange1/binsize+n33] + y1[binrange1+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        if n < maxbin3:
                            x[binrange1/binsize+binrange2/binsize+n33] = x[binrange1/binsize+binrange2/binsize+n33] + x1[binrange1+binrange2+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                            y[binrange1/binsize+binrange2/binsize+n33] = y[binrange1/binsize+binrange2/binsize+n33] + y1[binrange1+binrange2+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                    n22 = 0
                    n33 = n33 + 1
                    norm = 0.
            n = n + 1
    f.close()

    if rank == 0: print("n = ", n)

    y_SGC1 = np.arange(totbinrange,dtype=np.float)
    y_SGC = np.arange(totbinrange/binsize,dtype=np.float)
            
    n22 = 0
    n33 = 0
    norm = 0.
    with open(inputfile_SGC,"r") as f:
        n = 0
        for i in range(0, 31):
            next(f)
        for line in f:
            dummy2 = map(float, line.split())
            if n >= minbin1 and n < maxbin1:
                Nmodes[n-minbin1] = dummy2[6]
                x1[n-minbin1] = dummy2[0]
                y_SGC1[n-minbin1] = dummy2[2]
                x1[binrange1+n-minbin1] = dummy2[0]
                y_SGC1[binrange1+n-minbin1] = dummy2[3]
                if n < maxbin3:
                    x1[binrange1+binrange2+n-minbin1] = dummy2[0]
                    y_SGC1[binrange1+binrange2+n-minbin1] = dummy2[4]
                n22 = n22 + 1
                norm = norm + Nmodes[n-minbin1]
                if(n22 >= binsize):
                    x[n33] = 0
                    y_SGC[n33] = 0
                    x[binrange1/binsize+n33] = 0
                    y_SGC[binrange1/binsize+n33] = 0
                    if n < maxbin3:
                        x[binrange1/binsize+binrange2/binsize+n33] = 0
                        y_SGC[binrange1/binsize+binrange2/binsize+n33] = 0
                    for j in range(0, binsize):
                        x[n33] = x[n33] + x1[n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        y_SGC[n33] = y_SGC[n33] + y_SGC1[n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        x[binrange1/binsize+n33] = x[binrange1/binsize+n33] + x1[binrange1+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        y_SGC[binrange1/binsize+n33] = y_SGC[binrange1/binsize+n33] + y_SGC1[binrange1+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                        if n < maxbin3:
                            x[binrange1/binsize+binrange2/binsize+n33] = x[binrange1/binsize+binrange2/binsize+n33] + x1[binrange1+binrange2+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                            y_SGC[binrange1/binsize+binrange2/binsize+n33] = y_SGC[binrange1/binsize+binrange2/binsize+n33] + y_SGC1[binrange1+binrange2+n-minbin1-j]*Nmodes[n-minbin1-j]/norm
                    n22 = 0
                    n33 = n33 + 1
                    norm = 0.
            n = n + 1
    f.close()

    print("n = ", n)
            
    for i in range(0, totbinrange/binsize):
        print("k = ", x[i], " ps = ", y_SGC[i])
        print("k = ", x[i], " ps = ", y[i])

#################################

    C = np.zeros((totbinrange/binsize,totbinrange/binsize))
    C_SGC = np.zeros((totbinrange/binsize,totbinrange/binsize))

    yerr_NGC = np.arange(totbinrange/binsize,dtype=np.float)
    yerr_SGC = np.arange(totbinrange/binsize,dtype=np.float)

    cov_inputfile = "/Users/fbeutler/clustools/output/nbar_combined_output/covall_hex_combined_patchy_COMPnbar_%s_NGC_%d_%d_%d_%d_%d_%d_%d_%d.dat" % (tag2, minbin1/binsize, maxbin1/binsize, minbin2/binsize, maxbin2/binsize, minbin3/binsize, maxbin3/binsize, NGCmocks, kbins)
    cov_inputfile_SGC = "/Users/fbeutler/clustools/output/nbar_combined_output/covall_hex_combined_patchy_COMPnbar_%s_SGC_%d_%d_%d_%d_%d_%d_%d_%d.dat" % (tag2, minbin1/binsize, maxbin1/binsize, minbin2/binsize, maxbin2/binsize, minbin3/binsize, maxbin3/binsize, SGCmocks, kbins)

    with open(cov_inputfile) as fcov:
    
        for i in range(0, 4):
            next(fcov)
        n = 0
        n1 = 0
        for line in fcov:
            dummy = map(float, line.split())
            C[n][n1] = dummy[4]
            n = n + 1
            if n == totbinrange/binsize:
                n = 0
                n1 = n1 + 1

    fcov.close()
    print("n = ", n, ", n1 = ", n1)

    with open(cov_inputfile_SGC) as fcov:
    
        for i in range(0, 4):
            next(fcov)
        n = 0
        n1 = 0
        for line in fcov:
            dummy = map(float, line.split())
            C_SGC[n][n1] = dummy[4]
            n = n + 1
            if n == totbinrange/binsize:
                n = 0
                n1 = n1 + 1

    fcov.close()
    print("n = ", n, ", n1 = ", n1)

    Cinv = (NGCmocks - len(y) - 2)*inv(C)/(NGCmocks - 1)
    Cinv_SGC = (SGCmocks - len(y_SGC) - 2)*inv(C_SGC)/(SGCmocks - 1)
    if rank == 0: print(np.mat(Cinv)*np.mat(C))

    for i in range(0, totbinrange/binsize):
        yerr_NGC[i] = np.sqrt(C[i][i])
        yerr_SGC[i] = np.sqrt(C_SGC[i][i])
        print(yerr_NGC[i], yerr_SGC[i])

#################################
## Setting up the fit ###########
#################################

    free_name=[]

    ndim = 0
    for i in range(0, len(free_para)):
        if free_para[i] == True:
            free_name.append(all_name[i])
            ndim += 1
#################################
## Find maximum likelihood ######
#################################

    print("Running maximum likelihood...")
    chi2 = lambda *args: -2 * lnlike(*args)
    result = op.minimize(chi2, [start], args=(x, y, y_SGC, binrange1/binsize, binrange2/binsize, binrange3/binsize, maxbin1/binsize, Cinv, Cinv_SGC, free_para, fix_para))
    alpha_perp_ml, alpha_para_ml, fsig8_ml, b1NGCsig8_ml, b1SGCsig8_ml, b2NGCsig8_ml, b2SGCsig8_ml, NNGC_ml, NSGC_ml, sigmavNGC_ml, sigmavSGC_ml = match_para(result["x"], free_para, fix_para)
    free_ml = result["x"]
    minchi2 = lnlike(result["x"],x, y, y_SGC, binrange1/binsize, binrange2/binsize, binrange3/binsize, maxbin1/binsize, Cinv, Cinv_SGC,free_para, fix_para)*(-2.)
    print("min chi2 = ", minchi2, " dof = ", 2.*len(y)-ndim)
    print("""Maximum likelihood result:
        alpha_perp = {0} (start: {1})
        alpha_para = {2} (start: {3})
        fsig8 = {4} (start: {5})
        b1NGCsig8 = {6} (start: {7})
        b1SGCsig8 = {8} (start: {9})
        b2NGCsig8 = {10} (start: {11})
        b2SGCsig8 = {12} (start: {13})
        NNGC = {14} (start: {15})
        NSGC = {16} (start: {17})
        sigmavNGC = {18} (start: {19})
        sigmavSGC = {20} (start: {21})
        """.format(alpha_perp_ml, 0, alpha_para_ml, 0, fsig8_ml, 0, b1NGCsig8_ml, 0, b1SGCsig8_ml, 0, b2NGCsig8_ml, 0, b2SGCsig8_ml, 0, NNGC_ml, 0, NSGC_ml, 0, sigmavNGC_ml, 0, sigmavSGC_ml, 0))
###################################
###################################
## run MCMC #######################
###################################
    '''
    # Initialize the MPI-based pool used for parallelization.
    pool = MPIPool()
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
    # Only the master process will read the code from here on
    ''' # Set up the sampler.
    pos=[]
    sampler=[]
    rstate = np.random.get_state()

    print("ndim = ", ndim, len(start))
    print("Nchains = ", Nchains)

    for jj in range(0, Nchains):
        f = open("%s/RSDfit_chain_COMPnbar_%d_%d_%d_%d_%d_%d_%s_%d_chain%d.dat" % (outpath, minbin1/binsize, maxbin1/binsize, minbin2/binsize, maxbin2/binsize, minbin3/binsize, maxbin3/binsize, tag, rank, jj) , "w")
        f.close()
        pos.append([free_ml + temperature*(2.*np.random.random_sample((ndim,))-1.)*start for i in range(nwalkers)])
        sampler.append(emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, y_SGC, binrange1/binsize, binrange2/binsize, binrange3/binsize, maxbin1/binsize, Cinv, Cinv_SGC, free_para, fix_para)))
        #print("start pos ",pos[jj])

    # Start MCMC
    print("Running MCMC...")

    withinchainvar = np.zeros((Nchains,ndim))
    meanchain = np.zeros((Nchains,ndim))
    scalereduction = np.arange(ndim,dtype=np.float)
    for jj in range(0, ndim):
        scalereduction[jj] = 2.

    itercounter = 0
    chainstep = minlength
    loopcriteria = 1
    while loopcriteria:
        
        itercounter = itercounter + chainstep
        print("chain length =",itercounter," minlength =",minlength)
        
        for jj in range(0, Nchains):
            # Since we write the chain to a file we could put storechain=False, but in that case
            # the function sampler.get_autocorr_time() below will give an error
            for result in sampler[jj].sample(pos[jj], iterations=chainstep, rstate0=rstate, storechain=True, thin=ithin):
                pos[jj] = result[0]
                chainchi2 = -2.*result[1]
                rstate = result[2]
                out = open("%s/RSDfit_chain_COMPnbar_%d_%d_%d_%d_%d_%d_%s_%d_chain%d.dat" % (outpath, minbin1/binsize, maxbin1/binsize, minbin2/binsize, maxbin2/binsize, minbin3/binsize, maxbin3/binsize, tag, rank, jj), "a")
                for k in range(pos[jj].shape[0]):
                    out.write("{0:4d} {1:s} {2:0.6f}\n".format(k, " ".join(map(str,pos[jj][k])), chainchi2[k]))
                out.close()
    
            # we do the convergence test on the second half of the current chain (itercounter/2)
            chainsamples = sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim))
            #print("len chain = ", chainsamples.shape)
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)
    
        scalereduction = gelman_rubin_convergence(withinchainvar, meanchain, itercounter/2, Nchains, ndim)
        print("scalereduction = ", scalereduction)
        
        loopcriteria = 0
        for jj in range(0, ndim):
            if np.absolute(1-scalereduction[jj]) > epsilon:
                loopcriteria = 1

        chainstep = ichaincheck

    print("Done.")
    
    # Close the processes.
    #pool.close()

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional vector.
    for jj in range(0, Nchains):
        print("Mean acceptance fraction for chain ", jj,": ", np.mean(sampler[jj].acceptance_fraction))
    # Estimate the integrated autocorrelation time for the time series in each parameter.
    for jj in range(0, Nchains):
        print("Autocorrelation time for chain ", jj,": ", sampler[jj].get_autocorr_time())

###################################
## Compute the quantiles ##########
###################################

    #samples=[]
    mergedsamples=[]

    for jj in range(0, Nchains):
        #samples.append(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
        mergedsamples.extend(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
    print("length of merged chain = ", sum(map(len,mergedsamples))/ndim)

    mcmc_array = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0], v[4]-v[1], v[1]-v[3]), zip(*np.percentile(mergedsamples, [15.86555, 50, 84.13445, 2.2775, 97.7225], axis=0)))

    print("mcmc_array = ", mcmc_array)

    all_mcmc_array=[]
    all_ml = np.arange(npara,dtype=np.float)

    free_ml = start

    counter = 0
    for i in range(0, npara):
        print("counter = ", counter, " free_para[i] = ", free_para[i])
        if free_para[i] == True:
            all_mcmc_array.append(mcmc_array[counter])
            all_ml[i] = free_ml[counter]
            counter += 1
        else:
            all_mcmc_array.append(fix_para[i], 0, 0, 0, 0)
            all_ml[i] = 0
    
    res = open("%s/result_COMPnbar_%d_%d_%d_%d_%d_%d_%s.dat" % (outpath, minbin1/binsize, maxbin1/binsize, minbin2/binsize, maxbin2/binsize, minbin3/binsize, maxbin3/binsize, tag), "w")
    res.write("%f %d %d\n" % (minchi2, 2.*len(y), ndim))
    
    print("""MCMC result: mean (+/-68%) (+/-95%) start""")
    for i in range(0, npara):
        print("""{0} = {1[0]} (+{1[1]} -{1[2]}) (+{1[3]} -{1[4]})""".format(all_name[i], all_mcmc_array[i]))
        res.write("%f %f %f %f %f %f\n" % (all_mcmc_array[i][0],all_mcmc_array[i][1],all_mcmc_array[i][2],all_mcmc_array[i][3],all_mcmc_array[i][4],all_ml[i]))

    res.close()

# to call the main() function to begin the program.
if __name__ == '__main__':
    main()
