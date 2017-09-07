'''

Testing the validity of the Gaussian functional form assumption 
of the Likelihood for galaxy clustering analyses.


'''
import numpy as np 
from scipy.stats import gaussian_kde as gkde
# -- local -- 
import data as Data


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
    if isinstance(i_bins, float): 
        i_bins = [i_bins]
    pdfs = []
    for i_bin in i_bins: 
        kern = gkde(X_w[i_bin,:]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        pdfs.append(kern.evaluate(x))
    return pdfs 


def whiten(X): 
    ''' Given data matrix X, use Choletsky decomposition of the 
    precision matrix in order to decorrelate (aka whiten) the data.  

    Note data matrix X should be N_k x N_mock where N_k = # of k modes
    and N_mock = # of mocks. 

    Returns X_w (the whitened matrix) and W (the whitening matrix) -- X_w = W.T * X
    '''
    C_x = np.cov(X)  # covariance matrix of X 
    invC_x = np.linalg.inv(C_x) # precision matrix of X 
    W = np.linalg.cholesky(invC_x) 

    # whitened data
    X_w = np.dot(W.T, X)
    return X_w, W 


def dataX(mock, ell=0, rebin=None): 
    ''' Construct data matrix X from P(k) measures of mock catalogs.

    X_i = P_i - < P > 

    X has N_k x N_mock dimensions. 
    '''
    pkay = Data.Pk() # read in P(k) data 
    n_mock = pkay._n_mock(mock) 
    for i in range(1, n_mock+1):  
        pkay.Read(mock, i, ell=ell) 
        if rebin is None: 
            k, pk = pkay.k, pkay.pk
        else: 
            k, pk, cnt = pkay.rebin(rebin)

        if i == 1: 
            pks = np.zeros((len(k), n_mock))
        pks[:, i-1] = pk 

    # calculate < P(k) > from the mock. This will serve as m(theta)
    mu_pk = np.sum(pks, axis=1)/np.float(n_mock)

    X = (pks.T - mu_pk).T 
    return X
