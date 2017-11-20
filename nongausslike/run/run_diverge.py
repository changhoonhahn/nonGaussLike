import sys as Sys
import os 
import time
import numpy as np 

from numpy.random import multivariate_normal as mvn 
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture as GMix
from sklearn.neighbors import KernelDensity as skKDE 

import env 
import util as UT 
import nongauss as NG 
#import paper as Pap
# -- plotting -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def diverge(obvs, diver, div_func='kl', Nref=1000, K=5, n_mc=10, n_comp_max=10, pk_mock='patchy.z1', NorS='ngc'):
    ''' calculate the divergences: 

    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)

    '''
    if isinstance(Nref, float): Nref = int(Nref)
    if diver not in ['ref', 'pX_gauss', 'pX_GMM', 'pX_KDE', 'pXi_ICA_GMM', 'pXi_ICA_KDE']: 
        raise ValueError
    # read in mock data X  
    if obvs == 'pk': 
        X_mock = NG.X_pk_all(pk_mock, NorS=NorS, sys='fc')
    elif obvs == 'gmf': 
        X_mock = NG.X_gmf_all()
    else: 
        raise ValueError("obvs = 'pk' or 'gmf'")  
    n_mock = X_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    X_mock_meansub, _ = NG.meansub(X_mock) # mean subtract
    X_w, W = NG.whiten(X_mock_meansub)
    X_ica, _ = NG.Ica(X_w)  # ICA transformation 

    if diver in ['pX_gauss', 'ref']: 
        C_X = np.cov(X_w.T) # covariance matrix
    elif diver == 'pX_GMM': # p(mock X) GMM
        gmms, bics = [], [] 
        for i_comp in range(1,n_comp_max+1):
            gmm = GMix(n_components=i_comp)
            gmm.fit(X_w) 
            gmms.append(gmm)
            bics.append(gmm.bic(X_w))
        ibest = np.array(bics).argmin() 
        kern_gmm = gmms[ibest]
    elif diver == 'pX_KDE': # p(mock X) KDE 
        t0 = time.time() 
        grid = GridSearchCV(skKDE(),
                {'bandwidth': np.linspace(0.1, 1.0, 30)},
                cv=10) # 10-fold cross-validation
        grid.fit(X_w)
        kern_kde = grid.best_estimator_
        dt = time.time() - t0 
        print('%f sec' % dt) 
    elif diver == 'pXi_ICA_GMM':
        # PI p(X^i_ICA) GMM
        kern_gmm_ica = [] 
        for ibin in range(X_ica.shape[1]): 
            gmms, bics = [], [] 
            for i_comp in range(1,n_comp_max+1):
                gmm = GMix(n_components=i_comp)
                gmm.fit(X_ica[:,ibin][:,None]) 
                gmms.append(gmm)
                bics.append(gmm.bic(X_ica[:,ibin][:,None]))
            ibest = np.array(bics).argmin() 
            kern_gmm_ica.append(gmms[ibest])
    elif diver == 'pXi_ICA_KDE':
        # PI p(X^i_ICA) KDE  
        kern_kde_ica = [] 
        for ibin in range(X_ica.shape[1]): 
            t0 = time.time() 
            grid = GridSearchCV(skKDE(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=10) # 10-fold cross-validation
            grid.fit(X_ica[:,ibin][:,None]) 
            kern_kde_ica.append(grid.best_estimator_) 
            dt = time.time() - t0 
            print('%f sec' % dt) 
    
    str_obvs = ''
    if obvs == 'pk': str_obvs = '.'+NorS

    if 'renyi' in div_func: 
        alpha = float(div_func.split(':')[-1])
        str_div = 'renyi'+str(alpha) 
    elif div_func == 'kl': 
        str_div = 'kl'
    str_comp = ''
    if 'GMM' in diver: str_comp = '.ncomp'+str(n_comp_max) 

    f_dat = ''.join([UT.dat_dir(), 'diverg.', obvs, str_obvs, '.', diver, '.K', str(K), str_comp, 
        '.Nref', str(Nref), '.', str_div, '.dat'])
    if os.path.isfile(f_dat): 
        f = open(f_dat, 'a') 
    else: 
        f = open(f_dat, 'w') 

    # caluclate the divergences now 
    divs = []
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        if diver == 'pX_gauss': 
            # estimate divergence between gmfs_white and a 
            # Gaussian distribution described by C_gmf
            div_i = NG.kNNdiv_gauss(X_w, C_X, Knn=K, div_func=div_func, Nref=Nref)
        elif diver == 'ref': 
            # reference divergence in order to showcase the estimator's scatter
            # Gaussian distribution described by C_gmf with same n_mock mocks 
            gauss = mvn(np.zeros(X_mock.shape[1]), C_X, size=n_mock)
            div_i = NG.kNNdiv_gauss(gauss, C_X, Knn=K, div_func=div_func, Nref=Nref)
        elif diver == 'pX_GMM': # D( mock X || p(X) GMM)
            div_i = NG.kNNdiv_Kernel(X_w, kern_gmm, Knn=K, div_func=div_func, 
                    Nref=Nref, compwise=False) 
        elif diver == 'pX_KDE': # D( mock X || p(X) KDE)
            div_i = NG.kNNdiv_Kernel(X_w, kern_kde, Knn=K, div_func=div_func, 
                    Nref=Nref, compwise=False) 
            divs.append(div_i)
        elif diver == 'pXi_ICA_GMM': # D( mock X || PI p(X^i_ICA) GMM), 
            div_i = NG.kNNdiv_Kernel(X_ica, kern_gmm_ica, Knn=K, div_func=div_func, 
                    Nref=Nref, compwise=True)
        elif diver == 'pXi_ICA_KDE': # D( mock X || PI p(X^i_ICA) KDE), 
            div_i = NG.kNNdiv_Kernel(X_ica, kern_kde_ica, Knn=K, div_func=div_func, 
                    Nref=Nref, compwise=True)
        f.write('%s \n' % div_i)  
    f.close() 
    return None


if __name__=="__main__": 
    # e.g. python run_diverge.py gmf kl pX_GMM 1000 10 10 20 
    # e.g. python run_diverge.py pk kl pX_GMM 1000 10 10 20 
    obvs = Sys.argv[1]
    div_func = Sys.argv[2]
    div = Sys.argv[3]
    if div not in ['ref', 'pX_gauss', 'pX_GMM', 'pX_KDE', 'pXi_ICA_GMM', 'pXi_ICA_KDE']: 
        raise ValueError
    Nref = int(Sys.argv[4])
    K = int(Sys.argv[5])
    n_mc = int(Sys.argv[6])
    if 'GMM' in div: 
        ncomp = int(Sys.argv[7]) 
    else: 
        ncomp = 10
    
    if obvs == 'pk': 
        diverge(obvs, div, div_func='kl', Nref=Nref, K=K, n_mc=n_mc, n_comp_max=ncomp, 
            pk_mock='patchy.z1', NorS='ngc')
    elif obvs == 'gmf': 
        diverge(obvs, div, div_func='kl', Nref=Nref, K=K, n_mc=n_mc, n_comp_max=ncomp) 
