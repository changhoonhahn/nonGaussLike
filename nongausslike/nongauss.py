'''

Testing the validity of the Gaussian functional form assumption 
of the Likelihood for galaxy clustering analyses.


'''
import numpy as np 
from scipy.stats import gaussian_kde as gkde
from scipy.stats import multivariate_normal as multinorm
from sklearn.decomposition import FastICA, PCA
# -- local -- 
import data as Data


def lnL_ica(pk_obv, Pk):
    ''' Given 'observed' P(k) and mock P(k) data, calculate the ICA log likelihood estimate.
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_w, W = whiten(X) # whitened data
    
    X_ica, W_ica = Ica(X_w) # get ICA transformation 
    
    if len(pk_obv.shape) == 1: 
        x_obv = np.dot(np.dot(W.T, pk_obv - mu_X), W_ica) # mean subtract, whiten, and ica transform observd pk
        p_Xica = 0. 
    else: 
        x_obv = np.zeros(pk_obv.shape)
        for i_obv in range(pk_obv.shape[1]):
            x_obv[:,i_obv] = np.dot(np.dot(W.T, pk_obv[:,i_obv] - mu_X), W_ica)
        p_Xica = np.zeros(pk_obv.shape[1]) 

    for i in range(X_ica.shape[0]):
        if len(pk_obv.shape) == 1: 
            x_obv_i = x_obv[i]
        else: 
            x_obv_i = x_obv[i,:]
        p_Xica += np.log(p_Xw_i(X_ica, i, x=x_obv_i))
    return p_Xica


def lnL_pca_gauss(pk_obv, Pk): 
    ''' Gaussian pseudo-likelihood calculated using PCA decomposition for 
    convenience. i.e. the data vector is decomposed in PCA components so that
    L = p(x_pca,0)p(x_pca,1)...p(x_pca,n). Each p(x_pca,i) is estimated using 
    N(0, sig_pca,i). *Note by construction this gives the same likelihood 
    as the Gaussian functional pseudo-likelihood.*
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_w, W = whiten(X) # whitened data
    
    X_pca, W_pca = Pca(X_w) # get PCA transformation 
    var_pca = np.zeros(X_pca.shape[0])
    for i in range(X_pca.shape[0]): 
        var_pca[i] = np.var(X_pca[i,:])
    cov = np.diag(var_pca)
    ggg = multinorm(np.zeros(len(mu_X)), cov) 
    
    if len(pk_obv.shape) == 1: 
        x_obv = np.dot(np.dot(W.T, pk_obv - mu_X), W_pca) # mean subtract, whiten, and ica transform observd pk
        p_Xpca = 0. 
    else: 
        x_obv = np.zeros(pk_obv.shape)
        for i_obv in range(pk_obv.shape[1]):
            x_obv[:,i_obv] = np.dot(np.dot(W.T, pk_obv[:,i_obv] - mu_X), W_pca)
        p_Xpca = np.zeros(pk_obv.shape[1]) 
    
    return np.log(ggg.pdf(x_obv.T))


def lnL_pca(pk_obv, Pk): 
    ''' Gaussian pseudo-likelihood calculated using PCA decomposition for 
    convenience. i.e. the data vector is decomposed in PCA components so that
    L = p(x_pca,0)p(x_pca,1)...p(x_pca,n). Each p(x_pca,i) is estimated by 
    gaussian KDE of the PCA transformed mock data. 
    '''
    X, mu_X = meansub(Pk) # mean subtract
    X_w, W = whiten(X) # whitened data
    
    X_pca, W_pca = Pca(X_w) # get PCA transformation 
    
    if len(pk_obv.shape) == 1: 
        x_obv = np.dot(np.dot(W.T, pk_obv - mu_X), W_pca) # mean subtract, whiten, and pca transform observd pk
        p_Xpca = 0. 
    else: 
        x_obv = np.zeros(pk_obv.shape)
        for i_obv in range(pk_obv.shape[1]):
            x_obv[:,i_obv] = np.dot(np.dot(W.T, pk_obv[:,i_obv] - mu_X), W_pca)
        p_Xpca = np.zeros(pk_obv.shape[1]) 

    for i in range(X_pca.shape[0]):
        if len(pk_obv.shape) == 1: 
            x_obv_i = x_obv[i]
        else: 
            x_obv_i = x_obv[i,:]
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


def Pca(X, whiten=False, **pca_kwargs): 
    ''' Given mean subtracted and whitened data (presumably non-whitened data should work), 
    input data should be in N_k x N_mock form. returns ICA transformed data and unmixing matrix.
    '''
    n_comp = X.shape[0] # number of ICA components 

    pca = PCA(n_components=n_comp, whiten=whiten, **pca_kwargs)
    pca.fit_transform(X.T)
    
    # X_pca = np.dot(X, pca.components_.T) 
    X_pca = pca.transform(X.T)
    
    return X_pca.T, pca.components_.T


def Ica(X, algorithm='deflation', whiten=False, **ica_kwargs): 
    ''' Given mean subtracted and whitened data (presumably non-whitened data should work), 
    input data should be in N_k x N_mock form. returns ICA transformed data and unmixing matrix.
    '''
    n_comp = X.shape[0] # number of ICA components 

    ica = FastICA(n_components=n_comp, algorithm=algorithm, whiten=whiten, **ica_kwargs)
    ica.fit_transform(X.T)
    
    # X_ica = np.dot(X, ica.components_.T) 
    X_ica = ica.transform(X.T)
    
    return X_ica.T, ica.components_.T


def p_Xwi_Xwj(X_w, ij_bins, x=np.linspace(-5., 5., 100), y=np.linspace(-5., 5., 100)):  
    '''
    '''
    xx, yy = np.meshgrid(x, y) 
    pos = np.vstack([xx.ravel(), yy.ravel()])
    
    pdfs = [] 
    for i in range(ij_bins.shape[1]): 
        ij = ij_bins[:,i]
        if ij[0] != ij[1]: 
            kern = gkde(np.vstack([X_w[ij[0],:], X_w[ij[1],:]])) # 2D gaussian KDE kernel using "rule of thumb" scott's rule. 
            pdfs.append(kern(pos))
        else: 
            pdfs.append(0.)
    return pdfs


def p_Xw_i(X_w, i_bins, x=np.linspace(-5., 5., 100)): 
    ''' Calculate the gaussian KDE of the pdf for Xw[ibins,:]
    '''
    if isinstance(i_bins, int): 
        i_bins = [i_bins]
    pdfs = []
    for i_bin in i_bins: 
        kern = gkde(X_w[i_bin,:]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
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
    C_x = np.cov(X)  # covariance matrix of X 
    if not hartlap: 
        invC_x = np.linalg.inv(C_x) # precision matrix of X 
    else: 
        # include Hartlap et al. (2007) factor (i.e. the hartlap factor) 
        invC_x = np.linalg.inv(C_x) 
        invC_x *= (float(X.shape[1]) - float(invC_x.shape[0]) - 2.)/(float(X.shape[1]) - 1.)

    if method == 'choletsky':
        W = np.linalg.cholesky(invC_x) 

        # whitened data
        X_w = np.dot(W.T, X)
    elif method == 'pca': 
        d, V = np.linalg.eigh(C_x)

        D = np.diag(1. / np.sqrt(d+1e-18))

        W = np.dot(np.dot(V, D), V.T) 
        
        X_w = np.dot(W.T, X) 

    return X_w, W 


def meansub(X): 
    ''' Given data matrix X (N_k x N_mock), subtract out the mean 
    '''
    # calculate < P(k) > from the mock. This will serve as m(theta)
    n_mock = X.shape[1]
    mu_X = np.sum(X, axis=1)/np.float(n_mock)

    Xp = (X.T - mu_X).T 
    
    return Xp, mu_X


def dataX(mock, ell=0, krange=None, rebin=None, sys=None, k_arr=False): 
    ''' Construct data matrix X from P(k) measures of mock catalogs.

    X_i = P_i - < P > 

    X has N_k x N_mock dimensions. 
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
            pks = np.zeros((len(k), n_mock))
        pks[:, i-1] = pk 
    if not k_arr:
        return pks
    else: 
        return pks, k
