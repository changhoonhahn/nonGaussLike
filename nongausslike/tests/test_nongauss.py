'''

Tests for each step of nongauss.py

'''
import numpy as np 
from scipy.stats import gaussian_kde as gkde
from scipy.stats import multivariate_normal as mGauss
# -- local -- 
import data as Data
import util as UT 
import nongauss as NG 
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


def GMF_p_Xw_i(name, ica=False, pca=False): 
    ''' Test the probability distribution function of each X_w^i
    component -- p(X_w^i). First compare the histograms of p(X_w^i) 
    with N(0,1). Then compare the gaussian KDE of p(X_w^i).
    '''
    gmf = NG.X_gmf(name)
    X, _ = NG.meansub(gmf)
    str_w = 'W'
    if ica and pca: 
        raise ValueError
    if ica: # ICA components
        # ICA components do not need to be Gaussian.
        # in fact the whole point of the ICA transform
        # is to capture the non-Gaussianity...
        X_white, _ = NG.whiten(X) # whitened data
        X_w, _ = NG.Ica(X_white) 
        str_w = 'ICA'
    if pca: # PCA components
        X_w, _ = NG.whiten(X, method='pca') # whitened data
        str_w = 'PCA'
    if not ica and not pca: # just whitened 
        X_w, W = NG.whiten(X) # whitened data
    
    # p(X_w^i) histograms
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i_bin in range(X_w.shape[1]): 
        p_X_w, edges = np.histogram(X_w[:,i_bin], normed=True)
        p_X_w_arr = UT.bar_plot(edges, p_X_w)
        sub.plot(p_X_w_arr[0], p_X_w_arr[1])
    x = np.arange(-5., 5.1, 0.1)
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$X_{'+str_w+'}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$P(X_{'+str_w+'})$', fontsize=25) 
    sub.legend(loc='upper right') 

    # p(X_w^i) gaussian KDE fits  
    pdfs = NG.p_Xw_i(X_w, range(X_w.shape[1]), x=x)

    sub = fig.add_subplot(122)
    for i_bin in range(X_w.shape[1]): 
        sub.plot(x, pdfs[i_bin])
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$X_{W}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$P(X_{W})$', fontsize=25) 
    sub.legend(loc='upper right') 

    str_ica, str_pca = '', ''
    if ica: 
        str_ica = '.ICA'
    if pca: 
        str_pca = '.PCA'

    f = ''.join([UT.fig_dir(), 'tests/test.GMF_p_Xw_i', str_pca, str_ica, '.', name, '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def lnL_sys(mock, ell=0, rebin=None, sys='fc'): 
    ''' Compare the pseudo gaussian L with no systematics, ICA L estimation with 
    no systematics, and ICA L estimation with fiber collisions.
    '''
    # Likelihood without systematics 
    Pk_nosys = NG.dataX(mock, ell=ell, rebin=rebin, sys=None)
    gauss = NG.lnL_pca_gauss(Pk_nosys, Pk_nosys)
    ica_nosys = NG.lnL_ica(Pk_nosys, Pk_nosys)
    
    # Likelihood with specified systematics 
    Pk_sys = NG.dataX(mock, ell=ell, rebin=rebin, sys=sys)
    ica_sys = NG.lnL_ica(Pk_sys, Pk_sys)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32
    sub.hist(gauss, bins=nbin, range=[-2.2*Pk_nosys.shape[0], -0.8*Pk_nosys.shape[0]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$; no sys.')
    sub.hist(ica_nosys, bins=nbin, range=[-2.2*Pk_nosys.shape[0], -0.8*Pk_nosys.shape[0]], 
            normed=True, alpha=0.75, label='ICA; no sys.')
    sub.hist(ica_sys, bins=nbin, range=[-2.2*Pk_nosys.shape[0], -0.8*Pk_nosys.shape[0]], 
            normed=True, alpha=0.75, label='ICA; w/ sys.')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk_nosys.shape[0], -0.5*Pk_nosys.shape[0]])
    sub.legend(loc='upper left', prop={'size': 20}) 

    if rebin is None: # save fig
        f = ''.join([UT.fig_dir(), 'tests/test.lnL_sys.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.lnL_sys.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def lnL(mock, ell=0, rebin=None, krange=None): 
    ''' Test the ICA likelihood estimation and pseudo gaussian likelihood 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    ica = NG.lnL_ica(Pk, Pk) 
    gauss = NG.lnL_pca_gauss(Pk, Pk)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32
    sub.hist(gauss, bins=nbin, range=[-2.2*Pk.shape[1], -0.8*Pk.shape[1]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$')
    sub.hist(ica, bins=nbin, range=[-2.2*Pk.shape[1], -0.8*Pk.shape[1]], 
            normed=True, alpha=0.75, label='ICA')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[1], -0.5*Pk.shape[1]])
    sub.legend(loc='upper left', prop={'size': 20}) 

    str_rebin = ''
    if rebin is not None: 
        str_rebin = '.rebin'+str(rebin)
    str_krange = ''
    if krange is not None: 
        str_krange = '.kmin'+str(krange[0])+'.kmax'+str(krange[1])
    
    f = ''.join([UT.fig_dir(), 'tests/test.lnL.', mock, '.ell', str(ell), 
        str_rebin, str_krange, '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def lnL_pca_gauss(mock, ell=0, krange=None, NorS='ngc'): 
    ''' ***TESTED***
    Test that lnL_pca_gauss is consistent with chi-squared calculated directly 
    from the mocks with the covariance matrix. 
    '''
    # Calculate the lnL_pca_gauss
    Pk = NG.X_pk(mock, ell=ell, krange=krange, NorS=NorS)
    dpk , mu_pk = NG.meansub(Pk) 
    pca_gauss = NG.lnL_pca_gauss(dpk, Pk)

    # calculate chi-squared 
    C_X = np.cov(Pk.T) 
    chi2 = np.zeros(Pk.shape[0])
    for i in range(Pk.shape[0]): 
        chi2[i] = -0.5 * np.sum(np.dot(dpk[i,:], np.linalg.solve(C_X, dpk[i,:].T)))
    
    offset = chi2 - pca_gauss
    print offset

    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32 
    sub.hist(pca_gauss, bins=nbin, normed=True, alpha=0.75, label='PCA Gaussian')
    sub.hist(chi2, bins=nbin, normed=True, alpha=0.75, label=r'$\chi^2$')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[1], 0.])
    sub.legend(loc='upper left', prop={'size': 20}) 

    f = ''.join([UT.fig_dir(), 'tests/test.lnL_pca_gauss_test.', mock, '.ell', str(ell), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def lnL_pca_kde(mock, ell=0, rebin=None, krange=None): 
    ''' ***TESTED: expectedly, more discrepant for low number of 
    mock catalogs. For Nseries monopole with 1000 mocks, no 
    significant discrepancy in the likelihood distribution 
    *** 
    Test whether or not the Gaussian KDE approximation of pdfs 
    is sufficiently accurate by comparing the likelihood estimated
    from NG.lnL_pca vs NG.lnL_pca_gauss. If they are highly 
    discrepant, then KDE estimate of the pdfs are not very accurate. 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    pca_gauss = NG.lnL_pca_gauss(Pk, Pk)
    pca_kde = NG.lnL_pca(Pk, Pk) 

    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32 
    sub.hist(pca_gauss, bins=nbin, range=[-2.2*Pk.shape[1], -0.5*Pk.shape[1]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$')
    sub.hist(pca_kde, bins=nbin, range=[-2.2*Pk.shape[1], -0.8*Pk.shape[1]], 
            normed=True, alpha=0.75, label='$\mathcal{L^\mathtt{pseudo}}$ KDE estimate')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[1], -0.5*Pk.shape[1]])
    sub.legend(loc='upper left', prop={'size': 20}) 

    if rebin is None: # save fig
        f = ''.join([UT.fig_dir(), 'tests/test.lnL_kde_test.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.lnL_kde_test.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def ica(mock, ell=0, rebin=None): 
    ''' *** TESTED *** 
    Test that the ICA works!
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X) # whitened data
    X_ica, W = NG.Ica(X_w)
    
    # compare covariance? 
    C_X = np.cov(X.T)
    C_Xica = np.cov(X_ica.T) 

    prettyplot()
    fig = plt.figure(figsize=(20, 8))
    sub = fig.add_subplot(121)
    im = sub.imshow(np.log10(C_X), interpolation='none')
    sub.set_title('log(Cov.) of Data')
    fig.colorbar(im, ax=sub) 

    sub = fig.add_subplot(122)
    im = sub.imshow(C_Xica, interpolation='none')
    fig.colorbar(im, ax=sub) 
    sub.set_title('Cov. of ICA transformed Data')
    # save fig
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.ICAcov.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.ICAcov.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None
   

def p_Xwi_Xwj_outlier(mock, ell=0, rebin=None, krange=None, ica=False, pca=False): 
    ''' Compare the joint pdfs of whitened X components (i.e. X_w^i, X_w^j)
    p(X_w^i, X_w^j) to p(X_w^i) p(X_w^j) in order to test the independence 
    argument. 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    if ica and pca: 
        raise ValueError
    if ica: # ICA components
        X_white, _ = NG.whiten(X) # whitened data
        X_w, _ = NG.Ica(X_white) 
    if pca: # PCA components
        X_w, _ = NG.whiten(X, method='pca') # whitened data
    if not ica and not pca: # just whitened 
        X_w, _ = NG.whiten(X, method='choletsky') # whitened data
    
    x, y = np.linspace(-5., 5., 50), np.linspace(-5., 5., 50)
    xx, yy = np.meshgrid(x,y)
    pos = np.vstack([xx.ravel(), yy.ravel()])
    
    ij_i, ij_j = np.meshgrid(range(X_w.shape[1]), range(X_w.shape[1]))
    ij = np.vstack([ij_i.ravel(), ij_j.ravel()])

    # joint pdfs of X_w^i and X_w^j estimated from mocks  
    # i.e. p(X_w^i, X_w^j)
    pdfs_2d = NG.p_Xwi_Xwj(X_w, ij, x=x, y=y)

    # p(X_w^i) * p(X_w^j) estimated from mocks
    pXwi = NG.p_Xw_i(X_w, range(X_w.shape[1]), x=x)
    pXwj = pXwi 

    # calculate L2 norm difference betwen joint pdf and 2d gaussian 
    chi2 = np.zeros(len(pdfs_2d))
    for i in range(len(pdfs_2d)): 
        if not isinstance(pdfs_2d[i], float): 
            pXwipXwj = np.dot(pXwi[ij[0,i]][:,None], pXwj[ij[1,i]][None,:]).T.flatten()
            chi2[i] = np.sum((pXwipXwj - pdfs_2d[i])**2)
    
    # ij values with the highest chi-squared
    ii_out = np.argsort(chi2)[-10:]
    inc = np.where(ij[0,ii_out] > ij[1,ii_out]) 

    prettyplot()
    fig = plt.figure(figsize=(len(inc[0])*10, 8))
    for ii, i_sort_i in enumerate(ii_out[inc]): 
        sub = fig.add_subplot(1, len(inc[0]), ii+1)
        # plot p(X_w^i) * p(X_w^j) 
        pXwipXwj = np.dot(pXwi[ij[0,i_sort_i]][:,None], pXwj[ij[1,i_sort_i]][None,:]).T
        sub.contourf(xx, yy, pXwipXwj, cmap='gray_r', levels=[0.05, 0.1, 0.15, 0.2])
    
        # p(X_w^i, X_w^j) 
        Z = np.reshape(pdfs_2d[i_sort_i], xx.shape)
        cs = sub.contour(xx, yy, Z, colors='k', linestyles='dashed', levels=[0.05, 0.1, 0.15, 0.2])
        cs.collections[0].set_label('$\mathtt{p(X_w^i, X_w^j)}$') 

        sub.set_xlim([-3., 3.])
        sub.set_xlabel('$\mathtt{X_w^{i='+str(ij[0,i_sort_i])+'}}$', fontsize=25)
        sub.set_ylim([-3., 3.])
        sub.set_ylabel('$\mathtt{X_w^{j='+str(ij[1,i_sort_i])+'}}$', fontsize=25)
        if ii == 0: 
            sub.legend(loc='upper right', prop={'size':25})
        else: 
            sub.set_yticklabels([])
    
    str_ica, str_pca = '', ''
    if ica: 
        str_ica = '.ICA'
    if pca: 
        str_pca = '.PCA'
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xwi_Xwj_outlier', str_ica, str_pca, '.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xwi_Xwj_outlier', str_ica, str_pca, '.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None


def p_Xw_i_MISE(mock, ell=0, rebin=None, krange=None, method='choletsky', b=0.1):
    ''' Examine the pdf of X_w^i components that deviate significantly from  
    N(0,1) based on MISE 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X, method=method) # whitened data
    
    # calculate the chi-squared values of each p(X_w^i)  
    x = np.arange(-5., 5.1, 0.1)
    mise = np.zeros(X_w.shape[1])
    for i_bin in range(X_w.shape[1]): 
        mise[i_bin] = NG.MISE(X_w[:,i_bin], b=b) 

    # plot the most discrepant components. 
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    i_sort = np.argsort(mise)
    print 'outlier bins = ', i_sort[-5:]
    print 'mise = ', mise[i_sort[-5:]]

    nbin = int(10./b)
    for i_bin in i_sort[-10:]: 
        hb_Xi, Xi_edges = np.histogram(X_w[:,i_bin], bins=nbin, range=[-5., 5.], normed=True) 
        p_X_w_arr = UT.bar_plot(Xi_edges, hb_Xi)
        sub.plot(p_X_w_arr[0], p_X_w_arr[1])

    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X^{i}_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X^{i}_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 
    
    str_rebin = ''
    if rebin is not None: 
        str_rebin = '.rebin'+str(rebin)

    f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i_outlier.', method, '.', mock, '.ell', str(ell), 
        str_rebin, '.b', str(b), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None


def p_Xw_i_outlier(mock, ell=0, rebin=None, krange=None, method='choletsky'):
    ''' Examine the pdf of X_w^i components that deviate significantly from  
    N(0,1) 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X, method=method) # whitened data
    
    # calculate the chi-squared values of each p(X_w^i)  
    x = np.arange(-5., 5.1, 0.1)
    chi2 = np.zeros(X_w.shape[1])
    for i_bin in range(X_w.shape[1]): 
        kern = gkde(X_w[:,i_bin]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        chi2[i_bin] = np.sum((UT.gauss(x, 1., 0.) - kern.evaluate(x))**2)/np.float(len(x))
    
    # plot the most discrepant components. 
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    i_sort = np.argsort(chi2)
    print 'outlier bins = ', i_sort[-5:]
    for i_bin in i_sort[-10:]: 
        kern = gkde(X_w[:,i_bin]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        sub.plot(x, kern.evaluate(x))
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X^{i}_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X^{i}_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 
    
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i_outlier.', method, '.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i_outlier.', method, '.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None


def p_Xw_i(mock, ell=0, rebin=None, krange=None, ica=False, pca=False): 
    ''' Test the probability distribution function of each X_w^i
    component -- p(X_w^i). First compare the histograms of p(X_w^i) 
    with N(0,1). Then compare the gaussian KDE of p(X_w^i).
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    str_w = 'W'
    if ica and pca: 
        raise ValueError
    if ica: # ICA components
        # ICA components do not need to be Gaussian.
        # in fact the whole point of the ICA transform
        # is to capture the non-Gaussianity...
        X_white, _ = NG.whiten(X) # whitened data
        X_w, _ = NG.Ica(X_white) 
        str_w = 'ICA'
    if pca: # PCA components
        X_w, _ = NG.whiten(X, method='pca') # whitened data
        str_w = 'PCA'
    if not ica and not pca: # just whitened 
        X_w, W = NG.whiten(X) # whitened data
    
    # p(X_w^i) histograms
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i_bin in range(X_w.shape[1]): 
        p_X_w, edges = np.histogram(X_w[:,i_bin], normed=True)
        p_X_w_arr = UT.bar_plot(edges, p_X_w)
        sub.plot(p_X_w_arr[0], p_X_w_arr[1])
    x = np.arange(-5., 5.1, 0.1)
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X_{'+str_w+'}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X_{'+str_w+'})}$', fontsize=25) 
    sub.legend(loc='upper right') 

    # p(X_w^i) gaussian KDE fits  
    pdfs = NG.p_Xw_i(X_w, range(X_w.shape[1]), x=x)

    sub = fig.add_subplot(122)
    for i_bin in range(X_w.shape[1]): 
        sub.plot(x, pdfs[i_bin])
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 

    str_ica, str_pca = '', ''
    if ica: 
        str_ica = '.ICA'
    if pca: 
        str_pca = '.PCA'

    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i', str_pca, str_ica, '.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i', str_pca, str_ica, '.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def whiten_recon(mock, ell=0, rebin=None, krange=None, method='choletsky'): 
    ''' ***TESTED: The whitening matrices reconstruct the P(k)s*** 
    Test whether P(k) can be reconstructed using the whitening matrix  
    '''
    Pk, k = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange, k_arr=True)
    X, mu_X = NG.meansub(Pk)
    X_w, W = NG.whiten(X, method=method) # whitened data
    
    prettyplot()
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i in range(X.shape[0]): 
        sub.plot(k, Pk[i,:])
    if krange is None: 
        sub.set_xlim([1e-3, 0.5])
    else: 
        sub.set_xlim(krange)
    sub.set_xscale('log')
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([2e3, 2.5e5])
    
    np.random.seed(7)
    sub = fig.add_subplot(122)
    for i in range(X.shape[0]): 
        X_noise = np.random.normal(size=X_w.shape[1])
        X_rec = np.linalg.solve(W.T, X_noise.T)
        sub.plot(k, X_rec.T + mu_X)
    if krange is None: 
        sub.set_xlim([1e-3, 0.5])
    else: 
        sub.set_xlim(krange)
    sub.set_xscale('log')
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([2e3, 2.5e5])

    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten_recon.', method, '.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten_recon.', method, '.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def whiten(mock, ell=0, rebin=None, krange=None, method='choletsky'): 
    ''' ***TESTED: Choletsky decomposition fails for full binned Nseries
    P(k) because the precision matrix estimate is not positive definite***
    test the data whitening. 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X, method=method) # whitened data
    
    prettyplot()
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i in range(X.shape[1]): 
        sub.plot(range(X_w.shape[0]), X_w[:,i])
    
    sub.set_xlim([0, X.shape[0]]) 
    sub.set_xlabel('$\mathtt{k}$ bins', fontsize=25)
    sub.set_ylim([-7., 7.])
    sub.set_ylabel('$\mathtt{W^{T} (P^i_'+str(ell)+'- \overline{P_'+str(ell)+'})}$', fontsize=25)
    
    C_Xw = np.cov(X_w.T)
    sub = fig.add_subplot(122)
    im = sub.imshow(C_Xw, interpolation='none')
    fig.colorbar(im, ax=sub) 
    
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten.', method, '.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten.', method, '.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def dataX(mock, ell=0, rebin=None, krange=None): 
    ''' ***TESTED***
    Test the data X calculation 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin, krange=krange)
    X, _ = NG.meansub(Pk)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)

    for i in range(X.shape[0]): 
        sub.plot(range(X.shape[1]), X[i,:])
    
    sub.set_xlim([0, X.shape[1]]) 
    sub.set_xlabel('$\mathtt{k}$ bins', fontsize=25)
    sub.set_ylim([-1e5, 1e5])
    sub.set_ylabel('$\mathtt{P^i_'+str(ell)+'(k) - \overline{P_'+str(ell)+'(k)}}$', fontsize=25)
    if rebin is not None: 
        f = ''.join([UT.fig_dir(), 'tests/test.dataX.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.dataX.', mock, '.ell', str(ell), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def invC(mock, ell=0, rebin=None): 
    ''' ***TESTED***
    Test inverting the covariance matrix. This is
    inspired by the fact that the original binning of 
    Nseries P(k) is not invertible...
    '''
    pkay = Data.Pk()
    n_mock = pkay._n_mock(mock) 
    for i in range(1, n_mock+1):  
        pkay.Read(mock, i, NorS='ngc') 
        k, pk = pkay.k, pkay.pk

        if i == 1: 
            pks = np.zeros((len(k), n_mock))
        pks[:, i-1] = pk 

    C_pk = np.cov(pks.T) # covariance matrix 

    invC_pk = np.linalg.inv(C_pk) 

    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(1,2,1)
    im = sub.imshow(C_pk, interpolation='None')
    fig.colorbar(im, ax=sub)
    sub = fig.add_subplot(1,2,2)
    im = sub.imshow(invC_pk, interpolation='None')
    fig.colorbar(im, ax=sub)
    plt.show() 
    return None


if __name__=="__main__": 
    GMF_p_Xw_i('manodeep.run1', ica=True, pca=False)
    #lnL_pca_gauss('patchy.z1', ell=0, krange=[0.01, 0.15], NorS='ngc')
