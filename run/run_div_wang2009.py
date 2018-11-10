import sys 
import os 
import time
import numpy as np 
from numpy.random import multivariate_normal as mvn 
# -- nonGaussLike --
from nongausslike import util as UT 
from nongausslike import knn as kNN
from nongausslike import nongauss as NG 


def KL_wang(obvs, diver, Nref=1000, n_mc=10, n_comp_max=10, n_mocks=20000, 
        pk_mock='patchy.z1', NorS='ngc'):
    ''' calculate the divergences: 

    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    if isinstance(Nref, float): Nref = int(Nref)
    if diver not in ['ref', 'pX_gauss']: 
        raise ValueError
    str_obvs = ''
    if obvs == 'pk': str_obvs = '.'+NorS
    str_comp = ''
    if 'GMM' in diver: str_comp = '.ncomp'+str(n_comp_max) 

    f_dat = ''.join([UT.dat_dir(), 'diverg/', 
        'diverg.', obvs, str_obvs, '.', diver, str_comp, '.Nref', str(Nref), '.KLwang2009.dat'])

    if not os.path.isfile(f_dat): 
        print('-- writing to -- \n %s' % f_dat)
        f_out = open(f_dat, 'w') 
    else: 
        print('-- appending to -- \n %s' % f_dat)

    # read in mock data X  
    if obvs == 'pk': 
        X_mock = NG.X_pk_all(pk_mock, NorS=NorS, sys='fc')
    elif obvs == 'gmf': 
        if n_mocks is not None: 
            X_mock = NG.X_gmf_all()[:n_mocks]
        else: 
            X_mock = NG.X_gmf_all()
    else: 
        raise ValueError("obvs = 'pk' or 'gmf'")  
    n_mock = X_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    X_mock_meansub, _ = NG.meansub(X_mock) # mean subtract
    X_w, W = NG.whiten(X_mock_meansub)
    C_X = np.cov(X_w.T) # covariance matrix

    # caluclate the divergences now 
    divs = []
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        t0 = time.time() 
        Y = mvn(np.zeros(X_mock.shape[1]), C_X, size=Nref)
        if diver in ['pX_gauss', 'pX_gauss_hartlap']: 
            # estimate divergence between gmfs_white and a 
            # Gaussian distribution described by C_gmf
            div_i = kNN.KL_w2009_eq29(X_w, Y)
        elif diver == 'ref': 
            # reference divergence in order to showcase the estimator's scatter
            # Gaussian distribution described by C_gmf with same n_mock mocks 
            X_gauss = mvn(np.zeros(X_mock.shape[1]), C_X, size=n_mock)
            div_i = kNN.KL_w2009_eq29(X_gauss, Y)
        print(div_i)
        f_out = open(f_dat, 'a') 
        f_out.write('%f \n' % div_i)  
        f_out.close() 
    return None


def KL_wang_YX(obvs, diver, Nref=1000, n_mc=10, n_comp_max=10, n_mocks=20000, 
        pk_mock='patchy.z1', NorS='ngc'):
    ''' calculate the divergences: 

    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    if isinstance(Nref, float): Nref = int(Nref)
    if diver not in ['ref', 'pX_gauss']: 
        raise ValueError
    str_obvs = ''
    if obvs == 'pk': str_obvs = '.'+NorS
    str_comp = ''
    if 'GMM' in diver: str_comp = '.ncomp'+str(n_comp_max) 

    f_dat = ''.join([UT.dat_dir(), 'diverg/', 
        'diverg.YX.', obvs, str_obvs, '.', diver, str_comp, '.Nref', str(Nref), '.KLwang2009.dat'])

    if not os.path.isfile(f_dat): 
        print('-- writing to -- \n %s' % f_dat)
        f_out = open(f_dat, 'w') 
    else: 
        print('-- appending to -- \n %s' % f_dat)

    # read in mock data X  
    if obvs == 'pk': 
        X_mock = NG.X_pk_all(pk_mock, NorS=NorS, sys='fc')
    elif obvs == 'gmf': 
        if n_mocks is not None: 
            X_mock = NG.X_gmf_all()[:n_mocks]
        else: 
            X_mock = NG.X_gmf_all()
    else: 
        raise ValueError("obvs = 'pk' or 'gmf'")  
    n_mock = X_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    X_mock_meansub, _ = NG.meansub(X_mock) # mean subtract
    X_w, W = NG.whiten(X_mock_meansub)
    C_X = np.cov(X_w.T) # covariance matrix

    # caluclate the divergences now 
    divs = []
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        t0 = time.time() 
        Y = mvn(np.zeros(X_mock.shape[1]), C_X, size=Nref)
        if diver in ['pX_gauss', 'pX_gauss_hartlap']: 
            # estimate divergence between gmfs_white and a 
            # Gaussian distribution described by C_gmf
            div_i = kNN.KL_w2009_eq29(Y, X_w)
        elif diver == 'ref': 
            # reference divergence in order to showcase the estimator's scatter
            # Gaussian distribution described by C_gmf with same n_mock mocks 
            X_gauss = mvn(np.zeros(X_mock.shape[1]), C_X, size=n_mock)
            div_i = kNN.KL_w2009_eq29(Y, X_gauss)
        print(div_i)
        f_out = open(f_dat, 'a') 
        f_out.write('%f \n' % div_i)  
        f_out.close() 
    return None


if __name__=="__main__": 
    # e.g. python run_diverge.py gmf kl pX_GMM 1000 10 10 20 
    # e.g. python run_diverge.py pk kl pX_GMM 1000 10 10 20 
    obvs = sys.argv[1]
    div = sys.argv[2]
    xy_or_yx = sys.argv[3]
    Nref = int(sys.argv[4])
    n_mc = int(sys.argv[5])
    if 'GMM' in div: 
        ncomp = int(sys.argv[6]) 
    else: 
        ncomp = 10
    if obvs == 'pk': 
        if xy_or_yx == 'XY': 
            KL_wang(obvs, div, Nref=Nref, n_mc=n_mc, n_comp_max=ncomp, pk_mock='patchy.z1', NorS='ngc')
        elif xy_or_yx == 'YX': 
            KL_wang_yx(obvs, div, Nref=Nref, n_mc=n_mc, n_comp_max=ncomp, pk_mock='patchy.z1', NorS='ngc')
    elif obvs == 'gmf': 
        if xy_or_yx == 'XY': 
            KL_wang(obvs, div, Nref=Nref, n_mc=n_mc, n_comp_max=ncomp) 
        elif xy_or_yx == 'YX': 
            KL_wang_yx(obvs, div, Nref=Nref, n_mc=n_mc, n_comp_max=ncomp) 
