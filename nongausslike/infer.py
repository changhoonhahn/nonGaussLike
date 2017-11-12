import numpy as np 
import emcee
from emcee.utils import MPIPool

# --- local ---
import util as UT
import data as Dat 
import model as Mod
import nongauss as NG 


def W_importance(tag, chain, **kwargs): 
    ''' Given a dictionary with the MCMC chain, evaluate the likelihood ratio 
    '''
    if 'RSD' in tag: # Florian's RSD analysis 
        if 'zbin' not in kwargs.keys(): 
            raise ValueError('specify zbin in kwargs') 

        # read in BOSS P(k) (data D) 
        k_list, pk_ngc_data, pk_sgc_data = [], [], []
        pkay = Dat.Pk()
        for ell in [0,2,4]: 
            k, plk_ngc = pkay.Observation(ell, kwargs['zbin'], 'ngc')
            _, plk_sgc = pkay.Observation(ell, kwargs['zbin'], 'sgc')
            k_list.append(k)
            pk_ngc_data.append(plk_ngc)
            pk_sgc_data.append(plk_sgc)
        pk_ngc_data = np.concatenate(pk_ngc_data)
        pk_sgc_data = np.concatenate(pk_sgc_data)

        binrange1, binrange2, binrange3 = len(k_list[0]), len(k_list[1]), len(k_list[2])
        maxbin1 = len(k_list[0])+1
        k = np.concatenate(k_list)
        
        # calculate D - m(theta) for all the mcmc chain
        delta_ngc = chain['pk_ngc'] - pk_ngc_data
        delta_sgc = chain['pk_sgc'] - pk_sgc_data

        # import PATCHY mocks 
        pk_ngc_list, pk_sgc_list = [], [] 
        for ell in [0, 2, 4]:
            if ell == 4: kmax = 0.1 
            else: kmax = 0.15
            pk_ngc_list.append(NG.X_pk('patchy.z'+str(kwargs['zbin']), 
                krange=[0.01,kmax], ell=ell, NorS='ngc', sys='fc'))
            pk_sgc_list.append(NG.X_pk('patchy.z'+str(kwargs['zbin']), 
                krange=[0.01,kmax], ell=ell, NorS='sgc', sys='fc'))
        pk_ngc_mock = np.concatenate(pk_ngc_list, axis=1) 
        pk_sgc_mock = np.concatenate(pk_sgc_list, axis=1) 

        if tag == 'RSD_ica_pca': # P_ICA(D - m(theta)) / P_PCA(D - m(theta))
            lnP_ica_ngc = NG.lnL_ica(delta_ngc, pk_ngc_mock)
            lnP_pca_ngc = NG.lnL_pca(delta_ngc, pk_ngc_mock)
            
            lnP_ica_sgc = NG.lnL_ica(delta_sgc, pk_sgc_mock) 
            lnP_pca_sgc = NG.lnL_pca(delta_sgc, pk_sgc_mock) 

            lnP_den = lnP_pca_ngc + lnP_pca_sgc
            lnP_num = lnP_ica_ngc + lnP_ica_sgc
        elif tag == 'RSD_pca_gauss': # P_PCA,KDE (D - m(theta)) / P_Gauss(D - m(theta))
            lnP_pca_ngc = NG.lnL_pca(delta_ngc, pk_ngc_mock)
            lnP_gauss_ngc = NG.lnL_pca_gauss(delta_ngc, pk_ngc_mock) 
            
            lnP_pca_sgc = NG.lnL_pca(delta_sgc, pk_sgc_mock) 
            lnP_gauss_sgc = NG.lnL_pca_gauss(delta_sgc, pk_sgc_mock) 

            lnP_den = lnP_gauss_ngc + lnP_gauss_sgc
            lnP_num = lnP_pca_ngc + lnP_pca_sgc
        elif tag == 'RSD_ica_gauss': # P_ICA(D - m(theta)) / P_PCA,Gauss(D - m(theta))
            lnP_ica_ngc = NG.lnL_ica(delta_ngc, pk_ngc_mock) 
            lnP_gauss_ngc = NG.lnL_pca_gauss(delta_ngc, pk_ngc_mock) 
            
            lnP_ica_sgc = NG.lnL_ica(delta_sgc, pk_sgc_mock) 
            lnP_gauss_sgc = NG.lnL_pca_gauss(delta_sgc, pk_sgc_mock) 
            
            lnP_den = lnP_gauss_ngc + lnP_gauss_sgc
            lnP_num = lnP_ica_ngc + lnP_ica_sgc
        elif tag == 'RSD_ica_chi2': # P_ICA(D - m(theta)) / P_PCA,Gauss(D - m(theta))
            lnP_ica_ngc = NG.lnL_ica(delta_ngc, pk_ngc_mock) 
            lnP_ica_sgc = NG.lnL_ica(delta_sgc, pk_sgc_mock) 

            lnP_den = -0.5 * chain['chi2']
            lnP_num = lnP_ica_ngc + lnP_ica_sgc

    elif 'gmf' in tag: # GMF
        geemf = Dat.Gmf() # read in SDSS GMF (data D) 
        nbins, gmf_data = geemf.Observation()
        
        # calculate D - m(theta) for all the mcmc chain
        dgmf = gmf_data - chain['gmf'] 
        
        # read mock gmfs (all mocks from 100 differnet HOD parameter points)
        gmf_mock = NG.X_gmf_all() #gmf_mock = NG.X_gmf('manodeep.run'+str(kwargs['run']))#
        
        if tag == 'gmf_ica_chi2':
            lnP_den = -0.5 * chain['chi2'] # -0.5 chi-squared
            lnP_num = NG.lnL_ica(dgmf, gmf_mock)
        elif tag == 'gmf_all_chi2': 
            # importance weight determined by the ratio of 
            # the chi^2 from the chain and the chi^2 calculated 
            # using the covariance matrix from the entire catalog
            # we note that Sinha et al.(2017) does not include the
            # hartlap factor
            Cgmf = np.cov(gmf_mock.T) # covariance matrix 
            Cinv = np.linalg.inv(Cgmf) # precision matrix
             
            lnP_num = -0.5 * chain['chi2'] # chi-squared calculated 
            lnP_den = np.empty(dgmf.shape[0])
            for i in range(dgmf.shape[0]): # updated chi-square
                lnP_den[i] = -0.5 * np.dot(dgmf[i,:], np.dot(Cinv, dgmf[i,:].T)) 
        elif tag == 'gmf_pca_chi2':
            lnP_num = NG.lnL_pca(dgmf, gmf_mock)
            lnP_den = -0.5 * chain['chi2']
        else: 
            raise NotImplementedError
    else: 
        raise ValueError

    ws = np.exp(lnP_den - lnP_num)
    return [lnP_den, lnP_num, ws]


def mcmc_chains(tag, ichain=None): 
    ''' Given some tag string return mcmc chain in a dictionary 
    '''
    chain_dict = {} 
    if tag == 'beutler_z1': 
        # read in Florian's RSD MCMC chains  
        chain_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
            'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.dat']) 
        chain = np.loadtxt(chain_file, skiprows=1)

        labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC', 
                'N_NGC', 'N_SGC', 'sigmav_NGC', 'sigmav_SGC', 'chi2'] 
        for i in range(len(labels)): 
            chain_dict[labels[i]] = chain[:,i+1]

        # read in mock evaluations for the chains 
        chain_model_ngc_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
            'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.model.ngc.dat']) 
        chain_model_sgc_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
            'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.model.sgc.dat']) 
        pk_ngc = np.loadtxt(chain_model_ngc_file) 
        pk_sgc = np.loadtxt(chain_model_sgc_file) 
        chain_dict['pk_ngc'] = pk_ngc[:,1:]
        chain_dict['pk_sgc'] = pk_sgc[:,1:]

    elif tag == 'manodeep': 
        # read in Manodeep's GMF MCMC chains 
        chain_file = ''.join([UT.dat_dir(), 'manodeep/', 
            'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0.out']) 
        chain = np.loadtxt(chain_file, skiprows=5)
        labels = ['logMmin', 'sig_logM', 'logM0', 'logM1', 'alpha', 'chi2']
        
        for i,lbl in enumerate(labels): 
            if lbl == 'chi2': chain_dict[lbl] = -2.*chain[:,i]
            else: chain_dict[lbl] = chain[:,i]
        chain_dict['gmf'] = chain [:,-8:]
    else: 
        raise ValueError
    return chain_dict 


def mcmc(tag=None, zbin=1, nwalkers=48, Nchains=4, minlength=600, likelihood='pseudo'): 
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
    pkay = Dat.Pk()
    k0, p0k_ngc = pkay.Observation(0, zbin, 'ngc')
    k2, p2k_ngc = pkay.Observation(2, zbin, 'ngc')
    k4, p4k_ngc = pkay.Observation(4, zbin, 'ngc')
    pk_ngc_list = [p0k_ngc, p2k_ngc, p4k_ngc]
    k_list = [k0, k2, k4]
    # read in BOSS P(k) SGC
    k0, p0k_sgc = pkay.Observation(0, zbin, 'sgc')
    k2, p2k_sgc = pkay.Observation(2, zbin, 'sgc')
    k4, p4k_sgc = pkay.Observation(4, zbin, 'sgc')
    pk_sgc_list = [p0k_sgc, p2k_sgc, p4k_sgc]
    
    if likelihood == 'psuedo': # standard pseudo Gaussian likelihood 
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
   
        # ln Posterior function 
        lnPost = lnPost_pseudo
        # args for ln Posterior function 
        # data ks, BOSS NGC P_l(k), BOSS SGC P_l(k), NGC precision matrix, SGC precision matrix 
        lnpost_args = (k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    elif likelihood in ['pca', 'ica'] : 
        # read in patchy mock P(k)s for ngc and sgc 
        pk_ngc_list, pk_sgc_list = [], [] 
        for ell in [0, 2, 4]:
            if ell == 4: kmax = 0.1 
            else: kmax = 0.15
            pk_ngc_list.append(NG.X_pk('patchy.z'+str(kwargs['zbin']), 
                krange=[0.01,kmax], ell=ell, NorS='ngc', sys='fc'))
            pk_sgc_list.append(NG.X_pk('patchy.z'+str(kwargs['zbin']), 
                krange=[0.01,kmax], ell=ell, NorS='sgc', sys='fc'))
        pk_ngc_mock = np.concatenate(pk_ngc_list, axis=1) 
        pk_sgc_mock = np.concatenate(pk_sgc_list, axis=1) 
    else: 
        raise NotImplementedError
        
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
    
    print("initializing ", Nchains, " independent emcee chains")
    pos, samplers =[], []
    for ichain in range(Nchains):
        pos.append([start + temperature*start*(2.*np.random.random_sample(ndim)-1.)
            for i in range(nwalkers)])
        samplers.append(emcee.EnsembleSampler(nwalkers, ndim, lnPost, 
            args=lnpost_args, pool=pool)) 

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
                    output_str = '\t'.join(pos[jj][k].astype('str'))+'\t'+str(chainchi2[k])+'\n'
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


def lnLike_pca(theta, k_list, pk_ngc_list, pk_sgc_list, pk_ngc_mock, pk_sgc_mock):
    ''' wrapper for nongauss.lnL_pca . lnL_pca decomposes  P_data - P_model into
    PCA componenets, then measures the likelihood by calculating the probability
    of each of the components p(x_pca,i), which is estimated using a nonparametric
    density estimation method (e.g. KDE) from the mock catalogs 

    parameters
    ----------
    theta : array
        cosmological parameters : alpha_perp, alpha_para, fsig8 and 
        nuisance parameters : b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, 
        NNGC, NSGC, sigmavNGC, sigmavSGC

    k_list : list
        list of k values for the mono, quadru, and hexadecaopoles -- [k0, k2, k4]

    pk_ngc_list : list 
   
    pk_sgc_list : list 

    mocks_ngc : np.ndarray (N_mock x N_k)
        Array of the mock catalog P(k)s for NGC
    
    mocks_sgc : np.ndarray (N_mock x N_k)
        Array of the mock catalog P(k)s for SGC
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
        lnP_pca_ngc = NG.lnL_pca(delta_ngc, pk_ngc_mock) 
        lnP_pca_sgc = NG.lnL_pca(delta_sgc, pk_sgc_mock) 
        return lnP_pca_ngc + ln_pca_sgc #-0.5*(chi2_ngc + chi2_sgc)
    else: 
        return -0.5*(10000.)


def lnPost_pca(theta, k_list, pk_ngc_list, pk_sgc_list, pk_ngc_mock, pk_sgc_mock): 
    ''' log posterior 
    '''
    lp = lnPrior(theta)
    if np.isfinite(lp):
        return lp + lnLike_pca(theta, k_list, pk_ngc_list, pk_sgc_list, pk_ngc_mock, pk_sgc_mock)
    else:
        return -np.inf


def lnLike_pseudo(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc):
    ''' log of the pseudo Gaussian likelihood function. This is identical to 
    Florian's implementation. 

    parameters
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


def lnPost_pseudo(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc):
    ''' log posterior 
    '''
    lp = lnPrior(theta)
    if np.isfinite(lp):
        return lp + lnLike_pseudo(theta, k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    else:
        return -np.inf


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
