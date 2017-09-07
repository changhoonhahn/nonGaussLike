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


def p_Xwi_Xwj(mock, ell=0, rebin=None): 
    ''' Compare the joint pdfs of p(X_w^i, X_w^j) 
    '''
    X = NG.dataX(mock, ell=ell, rebin=rebin) # data matrix
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
    X = NG.dataX(mock, ell=ell, rebin=rebin) # data matrix
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
    X = NG.dataX(mock, ell=ell, rebin=rebin) # data matrix
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
    X = NG.dataX(mock, ell=ell, rebin=rebin) # data matrix
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
    X = NG.dataX(mock, ell=ell, rebin=rebin)
    
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
    p_Xwi_Xwj('qpm', ell=0, rebin=5)
