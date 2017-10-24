import numpy as np 
import emcee
from emcee.utils import MPIPool

# --- local ---
import util as UT
import data as Dat 
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
    if 0.8 < theta[0] < 1.4 and 0.8 < theta[1] < 1.4:
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
        return lp + lnLike(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    else:
        return -np.inf


#def importance_sampling():
#    ''' Use importance sampling in order to estimate the new posterior 
#    distribution. The mcmc chain is read in then weighted by the likelihood
#    ratio. 
#    '''
#    # read in MCMC chain 



def mcmc(tag=None, zbin=1, nwalkers=48, Nchains=4, minlength=600): 
    '''
    
    Parameters
    ---------- 

    Nchains : int 
        Number of independent chains to run for the gelman rubin convergence test
    
    '''
    if tag is None: 
        raise ValueError("specify a tag, otherwise it's confusing") 
    temperature = 2.e-3 # temperature

    # read in BOSS P(k) NGC
    k0, p0k_ngc = data(0, zbin, 'ngc')
    k2, p2k_ngc = data(2, zbin, 'ngc')
    k4, p4k_ngc = data(4, zbin, 'ngc')
    pk_ngc_list = [p0k_ngc, p2k_ngc, p4k_ngc]
    k_list = [k0, k2, k4]
    # read in BOSS P(k) SGC
    k0, p0k_sgc = data(0, zbin, 'sgc')
    k2, p2k_sgc = data(2, zbin, 'sgc')
    k4, p4k_sgc = data(4, zbin, 'sgc')
    pk_sgc_list = [p0k_sgc, p2k_sgc, p4k_sgc]

    # read in Covariance matrix 
    # currently for testing purposes, 
    # implemented to read in Florian's covariance matrix  
    _, _, C_pk_ngc = Dat.beutlerCov(zbin, NorS='ngc', ell='all')
    _, _, C_pk_sgc = Dat.beutlerCov(zbin, NorS='sgc', ell='all')

    # calculate precision matrices (including the hartlap factor) 
    Cinv_ngc = np.linalg.inv(C_pk_ngc)
    Cinv_sgc = np.linalg.inv(C_pk_sgc)
    # hartlap factor 
    n_mocks_ngc = 2045
    n_mocks_sgc = 2048
    f_hartlap_ngc = (float(n_mocks_ngc) - float(len(np.concatenate(pk_ngc_list))) - 2.)/(float(n_mocks_ngc) - 1.)
    f_hartlap_sgc = (float(n_mocks_sgc) - float(len(np.concatenate(pk_sgc_list))) - 2.)/(float(n_mocks_sgc) - 1.)
    Cinv_ngc *= f_hartlap_ngc  
    Cinv_sgc *= f_hartlap_sgc
        
    if zbin == 1: # 0.2 < z < 0.5 
        # maximum likelihood value 
        start = np.array([1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 0.32, -1580., -930., 6.1, 6.8] )
    ndim = len(start) 

    # initialize MPI pool
    try: 
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    except ValueError: 
        pool = None 

    # args for lnProb function 
    # data ks, BOSS NGC P_l(k), BOSS SGC P_l(k), NGC precision matrix, SGC precision matrix 
    lnpost_args = (k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    
    print("initializing ", Nchains, " independent emcee chains")
    pos, samplers =[], []
    for ichain in range(Nchains):
        pos.append([start + temperature*start*(2.*np.random.random_sample(ndim)-1.)
            for i in range(nwalkers)])
        samplers.append(emcee.EnsembleSampler(nwalkers, ndim, lnPost, args=lnpost_args, pool=pool)) 

    # Start MCMC
    print("Running MCMC...")
    withinchainvar = np.zeros((Nchains, ndim))
    meanchain = np.zeros((Nchains, ndim))
    scalereduction = np.repeat(2., ndim)
    
    # bunch of numbers for the mcmc run 
    itercounter = 0
    chainstep = minlength
    loop = 1
    epsilon = 0.02 #0.02
    ichaincheck = 100
    rstate = np.random.get_state()

    while loop:

        itercounter += chainstep
        print("chain length =",itercounter)

        for jj in range(Nchains):
            for result in samplers[jj].sample(pos[jj], iterations=chainstep, 
                    rstate0=rstate, storechain=True):
                pos[jj] = result[0]
                chainchi2 = -2.*result[1]
                rstate = result[2]

                # append chain outputs to chain file  
                chain_file = ''.join([UT.dat_dir(), 'mcmc/', tag, '.chain', str(jj), 
                    '.zbin', str(zbin), '.dat']) 
                f = open(chain_file, 'a')
                for k in range(pos[jj].shape[0]): 
                    output_str = '\t'.join(pos[jj][k].astype('str')) + '\n'
                    f.write(output_str)
                f.close()

            # we do the convergence test on the second half of the current chain (itercounter/2)
            chainsamples = samplers[jj].chain[:, itercounter/2:, :].reshape((-1, ndim))
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)
    
        scalereduction = gelman_rubin_convergence(withinchainvar, meanchain, itercounter/2, Nchains, ndim)
        print("scalereduction = ", scalereduction)
        
        loop = 0
        for jj in range(ndim):
            if np.abs(1-scalereduction[jj]) > epsilon:
                loopcriteria = 1

        chainstep = ichaincheck

    if pool is not None: 
        pool.close() 
    return None 


def gelman_rubin_convergence(withinchainvar, meanchain, n, Nchains, ndim):
    '''Calculate Gelman & Rubin diagnostic
     1. Remove the first half of the current chains
     2. Calculate the within chain and between chain variances
     3. estimate your variance from the within chain and between chain variance
     4. Calculate the potential scale reduction parameter
    '''
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


if __name__=="__main__":
    pass 
    #mcmc(tag='testing', zbin=1, nwalkers=48, Nchains=2, minlength=600)
