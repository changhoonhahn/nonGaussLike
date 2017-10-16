import numpy as np 
import emcee
from emcee.utils import MPIPool

# --- local ---
import util as UT
import model as Mod


def data(ell, zbin, nors): 
    ''' Read in P(k) measurements of BOSS from Florian  
    '''
    if ell == 0:
        str_pole = 'mono'
    elif ell == 2:
        str_pole = 'quadru'
    elif ell == 4:
        str_pole = 'hexadeca'
    str_pole += 'pole'

    fname = ''.join([UT.dat_dir(), 'Beutler/public_material_RSD/',
        'Beutleretal_pk_', str_pole, '_DR12_', nors.upper(), '_z', str(zbin), 
        '_prerecon_120.dat'])
    k_central, k_mean, pk = np.loadtxt(fname, skiprows=31, unpack=True, usecols=[0,1,2])
    return k_central, pk 


def lnPrior(theta):
    ''' log prior function -- ln(Prior(theta)). theta should have 
    [alpha_perp, alpha_para, fsig8, 
    b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, NNGC, NSGC, sigmavNGC, sigmavSGC]
    '''
    if len(theta) != 11: 
        raise ValueError("theta should have 11 parameters")

    #prior_min = [0.5, 0.5, 0.1, 0.3, 0.3, -6., -6., -10000., -10000., 0.5, 0.5]
    #prior_max = [1.5, 1.5, 1.1, 5., 5., 6., 6., 10000., 10000., 15., 15.]
    prior_min = [0.8, 0.8, 0.1, 0.3, 0.3, -6., -6., -10000., -10000., 0.5, 0.5]
    prior_max = [1.4, 1.4, 1.1, 5., 5., 6., 6., 10000., 10000., 15., 15.]

    for i in range(len(theta)): 
        if prior_min[i] < theta[i] < prior_max[i]: 
            pass
        else: 
            return -np.inf 
    return 0. 


def lnLike(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc):
    ''' log likelihood function 

    Parameters
    ----------
    theta : array
        cosmological parameters : alpha_perp, alpha_para, fsig8 and 
        nuisance parameters : b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, 
        NNGC, NSGC, sigmavNGC, sigmavSGC

    k_list : list
        list of k values for the mono, quadru, and hexadecaopoles -- [k0, k2, k4]

    pk_ngc_list : list 
        

    '''
    if theta[0] > 0.8 and theta[0] < 1.4 and theta[1] > 0.8 and theta[1] < 1.4:
        binrange1, binrange2, binrange3 = len(k_list[0]), len(k_list[1]), len(k_list[2])
        maxbin1 = len(k_list[0])+1

        k = np.concatenate(k_list)
        pk_ngc = np.concatenate(pk_ngc_list)
        pk_sgc = np.concatenate(pk_sgc_list)

        modelX = Mod.taruya_model(100, binrange1, binrange2, binrange3, maxbin1, 
                k, theta[0], theta[1], theta[2], theta[3], 
                theta[4], theta[5], theta[6], theta[7], 
                theta[8], theta[9], theta[10])
    
        model_ngc = modelX[0]
        model_sgc = modelX[1]
        #print("dummy_model_NGC = ", dummy_model_NGC, len(x))
        #for i in range(0, 37):
        #    print("k = ", x[i], dummy_model_NGC[i], dummy_model_SGC[i])

        diff_ngc = model_ngc - pk_ngc
        diff_sgc = model_sgc - pk_sgc 
        chi2_ngc = np.dot(diff_ngc, np.dot(Cinv_ngc, diff_ngc))
        chi2_sgc = np.dot(diff_sgc, np.dot(Cinv_sgc, diff_sgc))
        return -0.5*(chi2_ngc + chi2_sgc)
    else: 
        return -0.5*(10000.)


def lnPost(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc):
    ''' log posterior 
    '''
    lp = lnPrior(theta)
    if np.isfinite(lp):
        return lp + lnlike(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    else:
        return -np.inf


def mcmc(zbin=1, nwalkers=48, Nchains=4): 
    '''
    
    Parameters
    ---------- 

    Nchains : int 
        Number of independent chains to run for the gelman rubin convergence test
    
    '''
    temperature = 2.e-3 # temperature

    # read in BOSS P(k) NGC
    k0, p0k_ngc = Inf.data(0, zbin, 'ngc')
    k2, p2k_ngc = Inf.data(2, zbin, 'ngc')
    k4, p4k_ngc = Inf.data(4, zbin, 'ngc')
    pk_ngc_list = [p0k_ngc, p2k_ngc, p4k_ngc]
    k_list = [k0, k2, k4]
    # read in BOSS P(k) SGC
    k0, p0k_sgc = Inf.data(0, zbin, 'sgc')
    k2, p2k_sgc = Inf.data(2, zbin, 'sgc')
    k4, p4k_sgc = Inf.data(4, zbin, 'sgc')
    pk_sgc_list = [p0k_sgc, p2k_sgc, p4k_sgc]

    # read in Covariance matrix 
    
    if zbin == 1: # 0.2 < z < 0.5 
        # maximum likelihood value 
        start = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 0.32, -1580., -930., 6.1, 6.8] 
    ndim = len(start) 

    # initialize MPI pool
    #pool = MPIPool()
    #if not pool.is_master():
    #    pool.wait()
    #    sys.exit(0)

    # args for lnProb function 
    lnpost_args = (k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    
    print(Nchain, " independent emcee chains running")
    pos, samplers =[], []
    for ichain in range(Nchains):
        pos.append([start + temperature*start*(2.*np.random.random_sample(ndim)-1.)
            for i in range(nwalkers)])
        samplers.append(emcee.EnsembleSampler(nwalkers, ndim, lnPost, args=lnpost_args, pool=pool)) 

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

if __name__=="__main__":
    mcmc(zbin=1, nwalkers=4, Nchains=4)
