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
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


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


def lnL_b2017_krange(mock, ell=0): 
    ''' Test the ICA likelihood estimation and pseudo gaussian likelihood 
    within the k-range of the Beutler et al. (2017) 
    '''
    Pk = NG.dataX(mock, ell=ell, krange=[0.01, 0.15], rebin=5)
    ica = NG.lnL_ica(Pk, Pk) 
    gauss = NG.lnL_pca_gauss(Pk, Pk)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32
    sub.hist(gauss, bins=nbin, range=[-2.2*Pk.shape[0], -0.8*Pk.shape[0]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$')
    sub.hist(ica, bins=nbin, range=[-2.2*Pk.shape[0], -0.8*Pk.shape[0]], 
            normed=True, alpha=0.75, label='ICA')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[0], -0.5*Pk.shape[0]])
    sub.legend(loc='upper left', prop={'size': 20}) 
    # save fig
    f = ''.join([UT.fig_dir(), 'tests/test.lnL.b2017.krange.', mock, '.ell', str(ell), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 



def lnL(mock, ell=0, rebin=None): 
    ''' Test the ICA likelihood estimation and pseudo gaussian likelihood 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    ica = NG.lnL_ica(Pk, Pk) 
    gauss = NG.lnL_pca_gauss(Pk, Pk)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    if Pk.shape[1] < 100: 
        nbin = 20  
    else: 
        nbin = 32
    sub.hist(gauss, bins=nbin, range=[-2.2*Pk.shape[0], -0.8*Pk.shape[0]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$')
    sub.hist(ica, bins=nbin, range=[-2.2*Pk.shape[0], -0.8*Pk.shape[0]], 
            normed=True, alpha=0.75, label='ICA')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[0], -0.5*Pk.shape[0]])
    sub.legend(loc='upper left', prop={'size': 20}) 

    if rebin is None: # save fig
        f = ''.join([UT.fig_dir(), 'tests/test.lnL.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.lnL.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def lnL_pca_kde(mock, ell=0, rebin=None): 
    ''' ***TESTED: expectedly, more discrepant for low number of 
    mock catalogs. For Nseries monopole with 1000 mocks, no 
    significant discrepancy in the likelihood distribution 
    *** 
    Test whether or not the Gaussian KDE approximation of pdfs 
    is sufficiently accurate by comparing the likelihood estimated
    from NG.lnL_pca vs NG.lnL_pca_gauss. If they are highly 
    discrepant, then KDE estimate of the pdfs are not very accurate. 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    pca_gauss = NG.lnL_pca_gauss(Pk, Pk)
    pca_kde = NG.lnL_pca(Pk, Pk) 

    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    nbin = 32 
    sub.hist(pca_gauss, bins=nbin, range=[-2.2*Pk.shape[0], -0.5*Pk.shape[0]], 
            normed=True, alpha=0.75, label='Gaussian $\mathcal{L^\mathtt{pseudo}}$')
    sub.hist(pca_kde, bins=nbin, range=[-2.2*Pk.shape[0], -0.8*Pk.shape[0]], 
            normed=True, alpha=0.75, label='$\mathcal{L^\mathtt{pseudo}}$ KDE estimate')
    sub.set_xlabel('log $\mathcal{L}$', fontsize=25)
    sub.set_xlim([-2.2*Pk.shape[0], -0.5*Pk.shape[0]])
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

    X_ica, W = NG.ica(X_w)
    
    # compare covariance? 
    C_X = np.cov(X)
    C_Xica = np.cov(X_ica) 

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
    

def p_Xwi_Xwj(mock, ell=0, rebin=None): 
    ''' Compare the joint pdfs of p(X_w^i, X_w^j) 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X) # whitened data
    
    x, y = np.linspace(-5., 5., 50), np.linspace(-5., 5., 50)
    xx, yy = np.meshgrid(x,y)
    pos = np.vstack([xx.ravel(), yy.ravel()])
    
    # 2D gaussian 
    g2d = mGauss(np.array([0., 0.]), np.array([[1., 0.],[0., 1.]]))
    gauss2d = g2d.pdf(pos.T)
    
    ij_i, ij_j = np.meshgrid(range(X_w.shape[0]), range(X_w.shape[0]))
    ij = np.vstack([ij_i.ravel(), ij_j.ravel()])

    # joint pdfs of X_w^i and X_w^j estimated from mocks  
    pdfs_2d = NG.p_Xwi_Xwj(X_w, ij, x=x, y=y)
    # calculate L2 norm difference betwen joint pdf and 2d gaussian 
    chi2 = np.zeros(len(pdfs_2d))
    for i in range(len(pdfs_2d)): 
        if not isinstance(pdfs_2d[i], float): 
            chi2[i] = np.sum((gauss2d - pdfs_2d[i])**2)
    
    # ij values with the highest chi-squared
    ii_out = np.argsort(chi2)[-10:]
    inc = np.where(ij[0,ii_out] > ij[1,ii_out]) 

    prettyplot()
    fig = plt.figure(figsize=(len(inc[0])*10, 8))
    for ii, i_sort_i in enumerate(ii_out[inc]): 
        sub = fig.add_subplot(1, len(inc[0]), ii+1)
        # plot the reference 2D gaussian  
        sub.contourf(xx, yy, gauss2d.reshape(xx.shape), cmap='gray_r', levels=[0.05, 0.1, 0.15, 0.2])
    
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
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xwi_Xwj.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xwi_Xwj.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None


def Xw_i_outlier(mock, ell=0, rebin=None):
    ''' Examine the pdf of X_w^i components that deviate significantly from  
    N(0,1) 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X) # whitened data
    
    # calculate the chi-squared values of each p(X_w^i)  
    x = np.arange(-5., 5.1, 0.1)
    chi2 = np.zeros(X_w.shape[0])
    for i_bin in range(X_w.shape[0]): 
        kern = gkde(X_w[i_bin,:]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        chi2[i_bin] = np.sum((UT.gauss(x, 1., 0.) - kern.evaluate(x))**2)/np.float(len(x))
    
    # plot the most discrepant components. 
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    i_sort = np.argsort(chi2)
    print 'outlier bins = ', i_sort[-10:]
    for i_bin in i_sort[-10:]: 
        kern = gkde(X_w[i_bin,:]) # gaussian KDE kernel using "rule of thumb" scott's rule. 
        sub.plot(x, kern.evaluate(x))
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X^{i}_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X^{i}_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 
    
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i_outlier.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i_outlier.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None


def p_Xw_i(mock, ell=0, rebin=None): 
    ''' Test the probability distribution function of each X_w^i
    component -- p(X_w^i). First compare the histograms of p(X_w^i) 
    with N(0,1). Then compare the gaussian KDE of p(X_w^i).
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X) # whitened data
    
    prettyplot() 
    # p(X_w^i) histograms
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i_bin in range(X_w.shape[0]): 
        p_X_w, edges = np.histogram(X_w[i_bin,:], normed=True)
        p_X_w_arr = UT.bar_plot(edges, p_X_w)
        sub.plot(p_X_w_arr[0], p_X_w_arr[1])
    x = np.arange(-5., 5.1, 0.1)
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 

    # p(X_w^i) gaussian KDE fits  
    pdfs = NG.p_Xw_i(X_w, range(X_w.shape[0]), x=x)

    sub = fig.add_subplot(122)
    for i_bin in range(X_w.shape[0]): 
        sub.plot(x, pdfs[i_bin])
    sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=3, label='$\mathcal{N}(0,1)$')
    sub.set_xlim([-2.5, 2.5])
    sub.set_xlabel('$\mathtt{X_{W}}$', fontsize=25) 
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\mathtt{P(X_{W})}$', fontsize=25) 
    sub.legend(loc='upper right') 

    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.p_Xw_i.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def whiten(mock, ell=0, rebin=None): 
    ''' ***TESTED: Choletsky decomposition fails for full binned Nseries
    P(k) because the precision matrix estimate is not positive definite***
    test the data whitening. 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    X_w, W = NG.whiten(X) # whitened data
    
    prettyplot()
    fig = plt.figure(figsize=(15,7))
    sub = fig.add_subplot(121)
    for i in range(X.shape[1]): 
        sub.plot(range(X_w.shape[0]), X_w[:,i])
    
    sub.set_xlim([0, X.shape[0]]) 
    sub.set_xlabel('$\mathtt{k}$ bins', fontsize=25)
    sub.set_ylim([-7., 7.])
    sub.set_ylabel('$\mathtt{W^{T} (P^i_'+str(ell)+'- \overline{P_'+str(ell)+'})}$', fontsize=25)
    
    C_Xw = np.cov(X_w)
    sub = fig.add_subplot(122)
    im = sub.imshow(C_Xw, interpolation='none')
    fig.colorbar(im, ax=sub) 
    
    if rebin is None: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten.', mock, '.ell', str(ell), '.png'])
    else: 
        f = ''.join([UT.fig_dir(), 'tests/test.whiten.', mock, '.ell', str(ell), '.rebin', str(rebin), '.png'])
    fig.savefig(f, bbox_inches='tight') 
    return None 


def dataX(mock, ell=0, rebin=None): 
    ''' ***TESTED***
    Test the data X calculation 
    '''
    Pk = NG.dataX(mock, ell=ell, rebin=rebin)
    X, _ = NG.meansub(Pk)
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)

    for i in range(X.shape[1]): 
        sub.plot(range(X.shape[0]), X[:,i])
    
    sub.set_xlim([0, X.shape[0]]) 
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
        pkay.Read(mock, i) 
        k, pk = pkay.k, pkay.pk

        if i == 1: 
            pks = np.zeros((len(k), n_mock))
        pks[:, i-1] = pk 

    C_pk = np.cov(pks) # covariance matrix 

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
    lnL_b2017_krange('qpm', ell=0)
    lnL_b2017_krange('qpm', ell=2)
    lnL_b2017_krange('qpm', ell=4)
    #lnL_sys('nseries', ell=0, rebin=5)
    #lnL_sys('qpm', ell=0, rebin=None)
    #lnL_sys('nseries', ell=2, rebin=5)
    #lnL_sys('qpm', ell=2, rebin=5)
    #lnL_sys('nseries', ell=4, rebin=5)
    #lnL_sys('qpm', ell=4, rebin=5)
