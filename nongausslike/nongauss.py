'''

Testing the validity of the Gaussian functional form assumption 
of the Likelihood for galaxy clustering analyses.


'''
import numpy as np 
from scipy.stats import gaussian_kde as gkde
from scipy.stats import multivariate_normal as multinorm
from sklearn.decomposition import FastICA, PCA
# -- kNN divergence ---
from skl_groups.features import Features
from skl_groups.divergences import KNNDivergenceEstimator
# -- local -- 
import util as UT
import data as Data


def kNNdiv_ICA(X_white, X_ica, Knn=3, div_func='renyi:.5', Nref=None): 
    ''' `div_func` kNN divergence estimate between X_white and an ICA 
    distribution.
    '''
    if isinstance(Knn, int): 
        Knns = [Knn]
    elif isinstance(Knn, list): 
        Knns = Knn

    ica_dist = np.zeros((Nref, X_white.shape[1])) 
    for i_bin in range(X_ica.shape[1]): 
        kern = gkde(X_ica[:,i_bin]) # gaussian KDE kernel using "rule of thumb" scott's rule.
        ica_dist[:,i_bin] = kern.resample(Nref)
   
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


def lnL_ica(delta_pk, Pk):
    ''' Given 'observed' delta P(k) (a.k.a. P(k) observed - model P(k)) 
    and mock P(k) data, calculate the ICA log likelihood estimate.
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_w, W = whiten(X) # whitened data
    
    X_ica, W_ica = Ica(X_w) # get ICA transformation 
    
    if len(delta_pk.shape) == 1: 
        x_obv = np.dot(np.dot(delta_pk, W), W_ica) # mean subtract, whiten, and ica transform observd pk
        p_Xica = 0. 
    else: 
        x_obv = np.zeros(delta_pk.shape)
        for i_obv in range(delta_pk.shape[0]):
            x_obv[i_obv,:] = np.dot(np.dot(delta_pk[i_obv,:], W), W_ica)
        p_Xica = np.zeros(delta_pk.shape[0]) 

    for i in range(X_ica.shape[1]):
        if len(delta_pk.shape) == 1: 
            x_obv_i = x_obv[i]
        else: 
            x_obv_i = x_obv[:,i]
        p_Xica += np.log(p_Xw_i(X_ica, i, x=x_obv_i))
    return p_Xica


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


def lnL_pca(delta_pk, Pk): 
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
        p_Xpca = 0. 
    else: 
        x_obv = np.zeros(delta_pk.shape)
        for i_obv in range(delta_pk.shape[0]):
            x_obv[i_obv,:] = np.dot(delta_pk[i_obv,:], W_pca)
        p_Xpca = np.zeros(delta_pk.shape[0]) 

    for i in range(X_pca.shape[1]):
        if len(delta_pk.shape) == 1: 
            x_obv_i = x_obv[i]
        else: 
            x_obv_i = x_obv[:,i]
        p_Xpca += np.log(p_Xw_i(X_pca, i, x=x_obv_i))
    return p_Xpca


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


def p_Xwi_Xwj(X_w, ij_bins, x=np.linspace(-5., 5., 100), y=np.linspace(-5., 5., 100)):  
    ''' Calculate the gaussian KDE of the joint distribution Xwi and Xwj
    '''
    xx, yy = np.meshgrid(x, y) 
    pos = np.vstack([xx.ravel(), yy.ravel()])
    
    pdfs = [] 
    for i in range(ij_bins.shape[1]): 
        ij = ij_bins[:,i]
        if ij[0] != ij[1]: 
            # 2D gaussian KDE kernel using "rule of thumb" scott's rule. 
            kern = gkde(np.vstack([X_w[:,ij[0]], X_w[:,ij[1]]])) 
            pdfs.append(kern(pos))
        else: 
            pdfs.append(0.)
    return pdfs


def p_Xw_i(X_w, i_bins, x=np.linspace(-5., 5., 100)): 
    ''' Calculate the gaussian KDE of the pdf for Xw[:,ibins]
    '''
    if isinstance(i_bins, int): 
        i_bins = [i_bins]
    pdfs = []
    for i_bin in i_bins: 
        kern = gkde(X_w[:,i_bin]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        if len(i_bins) == 1: 
            return kern.evaluate(x)
        else: 
            pdfs.append(kern.evaluate(x))
    return np.array(pdfs)


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


def dataX(mock, ell=0, krange=None, rebin=None, sys=None, k_arr=False): 
    ''' Construct data matrix X from P(k) measures of mock catalogs.
    
    X_i = P(k)_i 

    X has N_mock x N_k dimensions. 
    '''
    pkay = Data.Pk() # read in P(k) data 
    n_mock = pkay._n_mock(mock) 
    for i in range(1, n_mock+1):  
        pkay.Read(mock, i, ell=ell, sys=sys) 
        pkay.krange(krange)
        if rebin is None: 
            k, pk = pkay.k, pkay.pk
        else: 
            k, pk, _ = pkay.rebin(rebin)

        if i == 1: 
            pks = np.zeros((n_mock, len(k)))
        pks[i-1,:] = pk 
    if k_arr:
        return pks, k
    return pks
