'''

Testing the validity of the Gaussian functional form assumption 
of the Likelihood for galaxy clustering analyses.


'''
import time
import numpy as np 
from scipy.stats import norm as Norm
from scipy.stats import gaussian_kde as gkde
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal as multinorm
from sklearn.neighbors import KernelDensity as skKDE 
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.mixture import GaussianMixture as GMix
from sklearn.decomposition import FastICA, PCA
# -- kNN divergence ---
from skl_groups.features import Features
from skl_groups.divergences import KNNDivergenceEstimator
# -- local -- 
import util as UT
import data as Data


def kNNdiv_Kernel(X_white, kernel, Knn=3, div_func='renyi:.5', Nref=None, compwise=True):  
    ''' `div_func` kNN divergence estimate between some data X_white and a distribution specified by Kernel.
    '''
    if isinstance(Knn, int): 
        Knns = [Knn]
    elif isinstance(Knn, list): 
        Knns = Knn
    # if component wise there should be X_white.shape[1]
    # kernels for each componenets 
    if compwise: 
        if X_white.shape[1] != len(kernel): raise ValueError
    
    # construct reference "bag"   
    if compwise: 
        ref_dist = np.zeros((Nref, X_white.shape[1])) 
        for icomp in range(X_white.shape[1]): 
            samp = kernel[icomp].sample(Nref)
            if isinstance(samp, tuple): 
                ref_dist[:,icomp] = samp[0].flatten()
            else: 
                ref_dist[:,icomp] = samp.flatten()
    else: 
        samp = kernel.sample(Nref)
        if isinstance(samp, tuple):  
            ref_dist = samp[0]
        else: 
            ref_dist = samp
    # estimate divergence  
    kNN = KNNDivergenceEstimator(div_funcs=[div_func], Ks=Knns, version='slow', clamp=False)
    feat = Features([X_white, ref_dist])
    div_knn = kNN.fit_transform(feat)
    if len(Knns) ==1: 
        return div_knn[0][0][0][1]
    div_knns = np.zeros(len(Knns))
    for i in range(len(Knns)): 
        div_knns[i] = div_knn[0][i][0][1]
    return div_knns


def kNNdiv_ICA(X_white, ica_kernel, Knn=3, div_func='renyi:.5', Nref=None, compwise=True, density_method='gkde', 
        n_comp_max=10): 
    ''' `div_func` kNN divergence estimate between some data X_white and an ICA 
    distribution.
    '''
    if isinstance(Knn, int): 
        Knns = [Knn]
    elif isinstance(Knn, list): 
        Knns = Knn

    if compwise: 
        if X_white.shape[1] != len(ica_kernel): raise ValueError

    if compwise: 
        ica_dist = np.zeros((Nref, X_ica.shape[1])) 
        for icomp in range(X_ica.shape[1]): 
            samp, _ = ica_kernel[icomp].sample(Nref)
            ica_dist[:,icomp] = samp
    else: 
        samp, _ = ica_kernel.sample(Nref)
        ica_dist = samp 
   
    kNN = KNNDivergenceEstimator(div_funcs=[div_func], Ks=Knns, version='slow', clamp=False)
    feat = Features([X_white, ica_dist])
    div_knn = kNN.fit_transform(feat)
    if len(Knns) ==1: 
        return div_knn[0][0][0][1]
    div_knns = np.zeros(len(Knns))
    for i in range(len(Knns)): 
        div_knns[i] = div_knn[0][i][0][1]
    return div_knns


def kNNdiv_gauss(X_white, cov_X, Knn=3, div_func='renyi:.5', gauss=None, Nref=None): 
    ''' `div_func` kNN divergence estimate between X_white and a 
    reference Gaussian with covariance matrix cov_X.
    '''
    if gauss is None: 
        if Nref is None: 
            raise ValueError
        gauss = np.random.multivariate_normal(np.zeros(X_white.shape[1]), cov_X, size=Nref) # Gaussian reference distribution
    if gauss.shape[1] != X_white.shape[1]:
        raise ValueError('dimension between X_white and Gaussian reference distribution do not match') 
    
    if isinstance(Knn, int): 
        Knns = [Knn]
    elif isinstance(Knn, list): 
        Knns = Knn

    kNN = KNNDivergenceEstimator(div_funcs=[div_func], Ks=Knns, version='slow', clamp=False)
    feat = Features([X_white, gauss])
    div_knn = kNN.fit_transform(feat)
    if len(Knns) ==1: 
        return div_knn[0][0][0][1]
    div_knns = np.zeros(len(Knns))
    for i in range(len(Knns)): 
        div_knns[i] = div_knn[0][i][0][1]
    return div_knns


def lnL_pX(delta_X, X_mock, density_method='gmm', n_comp_max=20, info_crit='bic', njobs=1): 
    ''' Given delta_X = observed X - model X, and data matrix
    X from mocks, estimate the log likelihood using a `density_method`
    (gmm or gkde) fit of p(X_mock).

    i.e. ln( p( X_obs - X_model(theta) | X_mock^(gmm/kde)) )
    '''
    if density_method not in ['gmm', 'kde']: raise ValueError 
    if len(delta_X.shape) == 1: 
        if len(delta_X) != X_mock.shape[1]: raise ValueError
    else: 
        if delta_X.shape[1] != X_mock.shape[1]: raise ValueError

    X, mu_X = meansub(X_mock)
    X_w, W = whiten(X) # whiten data 

    return lnp_Xw(X_w, x=delta_X, method=density_method, n_comp_max=n_comp_max, info_crit=info_crit, njobs=njobs)
    

def lnL_pXi_ICA(delta_X, X_mock, density_method='gmm', n_comp_max=20, info_crit='bic', njobs=1): 
    ''' Given delta_X = observed X - model X, and data matrix
    X from mocks, estimate the log likelihood using a `density_method`
    (gmm or gkde) fit of p(X_mock_i,ICA) for each componenet.
    
    ICA transformed X_mock
        X_ICA = X_mock x W_ica 
    ICA transformed delta X
        delta_X_ICA = delta_X x W_ica 

    ln( PI_i p( delta_X_ICA_i | X_ICA_i^(gmm/kde)) )
    '''
    X, mu_X = meansub(X_mock) # mean subtract
    X_w, W = whiten(X) # whitened data
    X_ica, W_ica = Ica(X_w) # get ICA transformation 

    if len(delta_X.shape) == 1: 
        x_obv = np.dot(np.dot(delta_X, W), W_ica) # whiten, and ica transform observd pk
        lnp_Xica = 0. 
    else: 
        x_obv = np.zeros(delta_X.shape)
        for i_obv in range(delta_X.shape[0]):
            x_obv[i_obv,:] = np.dot(np.dot(delta_X[i_obv,:], W), W_ica)
        lnp_Xica = np.zeros(delta_X.shape[0]) 

    for i in range(X_ica.shape[1]): # loop through each ICA component
        if len(delta_X.shape) == 1: 
            x_obv_i = x_obv[i]
        else: 
            x_obv_i = x_obv[:,i]
        lnp_Xica += lnp_Xw_i(X_ica, i, x=x_obv_i, method=density_method, n_comp_max=n_comp_max, 
                info_crit=info_crit, njobs=njobs)
    return lnp_Xica 


def lnL_ica(delta_pk, Pk, component_wise=True, density_method='gkde', n_comp_max=10):
    ''' Given 'observed' delta P(k) (a.k.a. P(k) observed - model P(k)) 
    and mock P(k) data, calculate the ICA log likelihood estimate.
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_w, W = whiten(X) # whitened data
    X_ica, W_ica = Ica(X_w) # get ICA transformation 
    
    if len(delta_pk.shape) == 1: 
        x_obv = np.dot(np.dot(delta_pk, W), W_ica) # whiten, and ica transform observd pk
        if component_wise: p_Xica = 0. 
    else: 
        x_obv = np.zeros(delta_pk.shape)
        for i_obv in range(delta_pk.shape[0]):
            x_obv[i_obv,:] = np.dot(np.dot(delta_pk[i_obv,:], W), W_ica)
        if component_wise: p_Xica = np.zeros(delta_pk.shape[0]) 

    if component_wise: 
        for i in range(X_ica.shape[1]):
            if len(delta_pk.shape) == 1: 
                x_obv_i = x_obv[i]
            else: 
                x_obv_i = x_obv[:,i]
            p_Xica += np.log(p_Xw_i(X_ica, i, x=x_obv_i, method=density_method, n_comp_max=n_comp_max))
    else: 
        p_Xica = np.log(p_Xw(X_ica, x=x_obv, method=density_method, n_comp_max=n_comp_max))
    return p_Xica


def lnL_pca(delta_pk, Pk, component_wise=True, density_method='gkde', n_comp_max=10): 
    ''' Gaussian pseudo-likelihood calculated using PCA decomposition -- 
    i.e. the data vector is decomposed in PCA components so that
    L = p(x_pca,0)p(x_pca,1)...p(x_pca,n). Each p(x_pca,i) is estimated by 
    gaussian KDE of the PCA transformed mock data. 
    
    Notes
    -----
    - This estimates the Gaussian functional pseudo-likelihood. Its main 
    use is to quantify the impact of the nonparametric density estimation 
    (e.g. the KDE).
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_pca, W_pca = whiten(X, method='pca') # whitened data
    
    if len(delta_pk.shape) == 1: 
        # PCA transform delta_pk 
        x_obv = np.dot(delta_pk, W_pca) 
        if component_wise: p_Xpca = 0. 
    else: 
        x_obv = np.zeros(delta_pk.shape)
        for i_obv in range(delta_pk.shape[0]):
            x_obv[i_obv,:] = np.dot(delta_pk[i_obv,:], W_pca)
        if component_wise: p_Xpca = np.zeros(delta_pk.shape[0]) 
    
    if component_wise: 
        for i in range(X_pca.shape[1]):
            if len(delta_pk.shape) == 1: 
                x_obv_i = x_obv[i]
            else: 
                x_obv_i = x_obv[:,i]
            p_Xpca += np.log(p_Xw_i(X_pca, i, x=x_obv_i, method=density_method, n_comp_max=n_comp_max))
    else: 
        p_Xpca = np.log(p_Xw(X_pca, x=x_obv, method=density_method, n_comp_max=n_comp_max))
    return p_Xpca


def lnL_pca_gauss(delta_pk, Pk): 
    ''' Gaussian pseudo-likelihood calculated using PCA decomposition. 
    i.e. the data vector is decomposed in PCA components so that
    L = p(x_pca,0)p(x_pca,1)...p(x_pca,n). Each p(x_pca,i) is estimated using 
    N(0, sig_pca,i). 
    
    Notes
    -----
    - By construction this evaluates the same likelihood as the 
    Gaussian functional pseudo-likelihood. It was implemented by convenience.
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_pca, W_pca = whiten(X, method='pca') # whitened data
    
    var_pca = np.zeros(X_pca.shape[1])
    for i in range(X_pca.shape[1]): 
        var_pca[i] = np.var(X_pca[:,i])
    cov = np.diag(var_pca)
    ggg = multinorm(np.zeros(len(mu_X)), cov) 
    
    if len(delta_pk.shape) == 1: 
        x_obv = np.dot(delta_pk, W_pca) 
    else: 
        x_obv = np.zeros(delta_pk.shape)
        for i_obv in range(delta_pk.shape[0]):
            x_obv[i_obv, :] = np.dot(delta_pk[i_obv,:], W_pca)
    return np.log(ggg.pdf(x_obv))


def lnL_gauss(pk_obv, Pk): 
    ''' *** Some silly issues with the offset that I don't want to deal with...***
    Given 'observed' P(k)  and mock P(k) data, calculate the log pseudo-likelihood 
    *with* Gaussian functional form assumption. 
    '''
    X, mu_X = meansub(Pk) # mean subtract

    C_X = np.cov(X) 
    
    ggg = multinorm(np.zeros(len(mu_X)), C_X)

    #offset = 0.5 * np.float(pk_obv.shape[0]) * np.log(2.*np.pi) + 0.5 * np.log(np.linalg.det(C_X))
    
    if len(pk_obv.shape) == 1: 
        return np.log(ggg.pdf(pk_obv - mu_X))
        #return -0.5 * np.sum(np.dot(x, np.linalg.solve(C_X, x))) - offset
    else: 
        lnL = np.zeros(pk_obv.shape[1])
        for i in range(pk_obv.shape[1]):
            lnL[i] = np.log(ggg.pdf(pk_obv[:,i] - mu_X))
            #lnL[i] = -0.5 * np.sum(np.dot(x, np.linalg.solve(C_X, x)))
        return lnL #- offset


def Ica(X, algorithm='deflation', whiten=False, **ica_kwargs): 
    ''' Given mean subtracted and whitened data (presumably non-whitened data should work), 
    input data should be in N_mock x N_k form. returns ICA transformed data and unmixing matrix.
    '''
    n_comp = X.shape[1] # number of ICA components 

    ica = FastICA(n_components=n_comp, algorithm=algorithm, whiten=whiten, **ica_kwargs)
    X_ica = ica.fit_transform(X)
    # X_ica = np.dot(X, ica.components_.T) 
    return X_ica, ica.components_.T


def MISE(Xis, b=0.1): 
    ''' Compare histogram of Xi's to normal distribution by calculating the 
    Mean integrated squared error (just L2) from Sellentin & Heavens. 
    '''
    nbin = int(10./b)
    hb_Xi, Xi_edges = np.histogram(Xis, bins=nbin, range=[-5., 5.], normed=True) 
    return np.sum((hb_Xi - UT.gauss(0.5*(Xi_edges[1:] + Xi_edges[:-1]), 1., 0.))**2)/np.float(nbin)


def p_Xw_i(X_w, i_bins, x=None, method='kde', n_comp_max=10): 
    ''' Estimate the pdf for Xw[:,ibins] at x using a nonparametric density estimation: 
    either gaussian KDE (gkde) or Gaussian Mixture Models (gmm)
    '''
    lnpdfs = lnp_Xw_i(X_w, i_bins, x=x, method=method, n_comp_max=n_comp_max)
    return [np.exp(lnpdf) for lnpdf in lnpdfs]


def lnp_Xw_i(X_w, i_bins, x=None, method='kde', n_comp_max=10, info_crit='bic', njobs=1):
    ''' Estimate the log pdf of X_w[:,i_bins] at x using a nonparametric 
    density estimation (either KDE or GMM). 
    
    parameters
    ----------
    X_w : np.ndarray 
        N_sample x N_feature matrix 

    i_bins : int or list of ints 
        specifies the feature bin(s) 

    x : np.ndarray or list of np.ndarray
        values to evaluate the pdf. Must be consistent with 
        i_bins!
    '''
    if x is None: raise ValueError
    if method not in ['kde', 'gmm']: raise ValueError("method = gkde or gmm") 
    if isinstance(i_bins, int): i_bins = [i_bins]
    if np.max(i_bins) > X_w.shape[1] or np.min(i_bins) < 0: raise ValueError
    if len(i_bins) > 1:  # more than one bin 
        if not isinstance(x, list): raise ValueError
        else: 
            if len(i_bins) != len(x): raise ValueError 
    else: x = [x] 

    pdfs = []
    for ii, i_bin in enumerate(i_bins): 
        if method == 'gmm': 
            # find best fit component using information criteria (BIC/AIC)
            gmms, ics = [], [] 
            for i_comp in range(1,n_comp_max+1):
                gmm = GMix(n_components=i_comp)
                gmm.fit(X_w[:,i_bin]) 
                gmms.append(gmm)
                if info_crit == 'bic':  # Bayesian Information Criterion
                    ics.append(gmm.bic(X_w[:,i_bin]))
                elif info_crit == 'aic': # Akaike information criterion
                    ics.append(gmm.aic(X_w[:,i_bin]))
            ibest = np.array(ics).argmin() # lower the better!
            kern = gmms[ibest]
        elif method == 'kde': 
            # find the best fit bandwidth using cross-validation grid search  
            t0 = time.time()
            grid = GridSearchCV(skKDE(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=10, n_jobs=njobs) # 10-fold cross-validation
            grid.fit(X_w[:,i_bin][:,None])
            kern = grid.best_estimator_
            dt = time.time() - t0 
            print('%f sec' % dt) 
        pdfs.append(kern.score_sample(x[ii][:,None]))  
    return pdfs 


def lnp_Xw(X_w, x=None, method='gmm', n_comp_max=10, info_crit='bic', njobs=1): 
    ''' Estimate the multi-dimensional pdf at x for a given X_w using a 
    nonparametric density estimation (either KDE or GMM). 
    '''
    if x is None: raise ValueError
    if method not in ['kde', 'gmm']: raise ValueError("method = gkde or gmm") 

    if method == 'gmm': 
        # find best fit component using information criteria (BIC/AIC)
        gmms, ics = [], [] 
        for i_comp in range(1,n_comp_max+1):
            gmm = GMix(n_components=i_comp)
            gmm.fit(X_w) 
            gmms.append(gmm)
            if info_crit == 'bic':  # Bayesian Information Criterion
                ics.append(gmm.bic(X_w))
            elif info_crit == 'aic': # Akaike information criterion
                ics.append(gmm.aic(X_w))
        ibest = np.array(ics).argmin() # lower the better!
        kern = gmms[ibest]
    elif method == 'kde': 
        # find the best fit bandwidth using cross-validation grid search  
        grid = GridSearchCV(skKDE(),
                {'bandwidth': np.linspace(0.1, 1.0, 30)},
                cv=10, njobs=njobs) # 10-fold cross-validation
        grid.fit(X_w)
        kern = grid.best_estimator_
    
    if len(x.shape) == 1: 
        return kern.score_samples(x[:,None]) 
    else: 
        return kern.score_samples(x) 


def GMM_pdf(gmm_obj): 
    ''' Return a function that will evaluate the pdf of the GMM 
    given sklearn.mixture.GaussianMixture object
    '''
    ws = gmm_obj.weights_
    mus = gmm_obj.means_
    cov = gmm_obj.covariances_

    ndim = mus[0].shape[0] # number of dimensions
    ncomp = mus.shape[0]

    def pdf(xx):  
        pdfx = 0.
        for icomp in range(ncomp):
            pdfx += ws[icomp]*multinorm.pdf(xx, mean=mus[icomp], cov=cov[icomp])
        return pdfx
    return pdf 


def whiten(X, method='choletsky', hartlap=False): 
    ''' Given data matrix X, use Choletsky decomposition of the 
    precision matrix in order to decorrelate (aka whiten) the data.  

    Note data matrix X should be N_k x N_mock where N_k = # of k modes
    and N_mock = # of mocks. 

    Returns X_w (the whitened matrix) and W (the whitening matrix) -- X_w = W.T * X
    '''
    C_x = np.cov(X.T)  # covariance matrix of X 

    invC_x = np.linalg.inv(C_x) # precision matrix of X 
    if hartlap: 
        # include Hartlap et al. (2007) factor (i.e. the hartlap factor) 
        invC_x *= (float(X.shape[0]) - float(invC_x.shape[1]) - 2.)/(float(X.shape[0]) - 1.)

    if method == 'choletsky':
        W = np.linalg.cholesky(invC_x) 
        return np.dot(X, W), W

    elif method == 'pca': 
        n_sample, n_comp = X.shape[0], X.shape[1] 
        pca = PCA(n_components=n_comp, whiten=True)
        X_w = pca.fit_transform(X)
        # X_w = np.dot(X, V)
        V = pca.components_.T / np.sqrt(pca.explained_variance_)
        return X_w, V

    elif method == 'stddev_scale':  
        # simplest whitening where each X k-bin is scaled by the 
        # standard deviation of that k-bin. 
        X_w = X 
        for i in range(X.shape[0]): 
            X_w[i,:] /= np.std(X[i,:])
        return X_w, None


def meansub(X): 
    ''' Given data matrix X (dimensions N_mock x N_k), subtract out the mean 
    '''
    # calculate < P(k) > from the mock. This will serve as m(theta)
    n_mock = X.shape[0]
    mu_X = np.sum(X, axis=0)/np.float(n_mock)
    return X - mu_X, mu_X


# --- data matrices from mocks --- 
def X_gmf_all(n_arr=False): 
    ''' Construct data matrix X from mock GMFs. But instead of 
    using one theta_HOD realization like the function X_gmf, use 
    all 100 different theta_HOD realizations.  
    '''
    X = [] 
    for i in range(100): 
        name = ('manodeep.run%i' % i) 
        if n_arr and i == 0: 
            X_i, nbins = X_gmf(name, n_arr=True) 
        else: 
            X_i = X_gmf(name) 
        X.append(X_i) 
    if n_arr:
        return np.concatenate(X), nbins
    return np.concatenate(X)  


def X_gmf(name, n_arr=False): 
    ''' Construct data matrix X from mock GMFs 
    
    X_i = gmf(N)_i      X has N_mock x N_n dimensions. 
    '''
    geemf = Data.Gmf() # read in GMF mocks 
    n_mock = geemf._n_mock(name) 
    for i in range(n_mock):  
        ibox = i % 4
        ireal = int((i - ibox)/4)+1
        geemf.Read(name, ireal, ibox)

        if i == 0: gmfs = np.zeros((n_mock, len(geemf.gmf)))
        gmfs[i,:] = geemf.gmf 

    if n_arr:
        return gmfs, geemf.nbins
    return gmfs 


def X_pk_all(mock, NorS='ngc', sys='fc', k_arr=False): 
    ''' Construct data matrix X from P(k) measures of mock catalogs. 
    
    X_i = [P_0(k)_i, P_2(k)_i, P_4(k)_i] 

    X has N_mock x [N_k0, N_k2, N_k4] dimensions. k ranges are equivalent
    to the k range of Beutler et al. (2017). 
    '''
    pk_list = []  
    for ell in [0, 2, 4]:
        if ell == 4: kmax = 0.1 
        else: kmax = 0.15
        pk = X_pk(mock, ell=ell, NorS=NorS, sys=sys, krange=[0.01,kmax])
        pk_list.append(pk.T)
    pk_all = np.concatenate(pk_list) 
    return pk_all.T


def X_pk(mock, ell=0, krange=None, NorS='ngc', sys='fc', k_arr=False): 
    ''' Construct data matrix X from P(k) measures of mock catalogs.
    
    X_i = P(k)_i 

    X has N_mock x N_k dimensions. 
    '''
    pkay = Data.Pk() # read in P(k) data 
    n_mock = pkay._n_mock(mock) 
    for i in range(1, n_mock+1):  
        pkay.Read(mock, i, ell=ell, NorS=NorS, sys=sys) 
        pkay.krange(krange)
        k, pk = pkay.k, pkay.pk

        if i == 1: 
            pks = np.zeros((n_mock, len(k)))
        pks[i-1,:] = pk 
    if k_arr:
        return pks, k
    return pks
