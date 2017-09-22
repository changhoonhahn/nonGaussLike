'''
'''
import numpy as np 

# --- local ---
import env
import util as UT
import data as Data

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def beutler_patchy_Cov(zbin, NorS='ngc'): 
    ''' compare my patchy covariance to Florian's 
    '''
    C_patchy = Data.patchyCov(zbin, NorS=NorS)
    C_beutler = Data.beutlerCov(zbin, NorS=NorS)
    
    prettyplot()
    fig = plt.figure(figsize=(21, 8))
    sub = fig.add_subplot(131)
    im = sub.imshow(np.log10(C_patchy), interpolation='none')
    sub.set_title('patchy log(Cov.)')
    fig.colorbar(im, ax=sub) 

    sub = fig.add_subplot(132)
    im = sub.imshow(np.log10(C_beutler), interpolation='none')
    sub.set_title('Beutler et al. log(Cov.)')
    fig.colorbar(im, ax=sub) 
    
    # normalized residual
    sub = fig.add_subplot(133)
    im = sub.imshow(np.abs((C_patchy - C_beutler)/C_beutler), interpolation='none', 
            vmin=0., vmax=1.)
    sub.set_title('Residual')
    fig.colorbar(im, ax=sub) 
    fig.savefig(''.join([UT.fig_dir(), 'Cov_beutler_patchy.z', str(zbin), '.', NorS, '.comparison.png']),
            bbox_inches='tight') 
    plt.close()
    return None 


def beutler_patchy_Cov_diag(zbin, NorS='ngc'): 
    ''' compare the diagonal elements of my patchy covariance to Florian's 
    '''
    C_patchy = Data.patchyCov(zbin, NorS=NorS, clobber=True)
    C_beutler = Data.beutlerCov(zbin, NorS=NorS)
    
    prettyplot()
    fig = plt.figure(figsize=(10, 8))
    sub = fig.add_subplot(111)
    
    sub.plot(range(C_patchy.shape[0]), C_patchy.diagonal()/C_beutler.diagonal())
    sub.plot(range(C_patchy.shape[0]), np.zeros(C_patchy.shape[0]), c='k', ls='--') 
    sub.vlines(14, -1, 2, color='k', linestyle='-', linewidth=3)
    sub.vlines(28, -1, 2, color='k', linestyle='-', linewidth=3)
    sub.set_xlim([0, C_patchy.shape[0]])
    sub.set_ylabel('$i$', fontsize=25) 
    sub.set_ylim([0.6, 1.2])
    sub.set_ylabel('$C^{patchy}_{i,i}/C^{Beutler}_{i,i}$', fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'Cov_ii_beutler_patchy..z', str(zbin), '.', NorS, '.comparison.png']),
            bbox_inches='tight') 
    plt.close()
    return None 



def patchyCov(zbin, NorS='ngc'): 
    '''***TESTED*** 
    Test patchy covariance matrix calcuation 
    '''
    C_X = Data.patchyCov(zbin, NorS=NorS)
    
    prettyplot()
    fig = plt.figure(figsize=(20, 8))
    sub = fig.add_subplot(111)
    im = sub.imshow(np.log10(C_X), interpolation='none')
    sub.set_title('patchy et al. log(Cov.)')
    fig.colorbar(im, ax=sub) 
    plt.show() 
    return None 


def beutlerCov(zbin, NorS='ngc'):
    ''' ***TESTED***
    Test reading in Florian's covariance matrix
    '''
    C_X = Data.beutlerCov(zbin, NorS=NorS)
    
    prettyplot()
    fig = plt.figure(figsize=(20, 8))
    sub = fig.add_subplot(111)
    im = sub.imshow(np.log10(C_X), interpolation='none')
    sub.set_title('Beutler et al. log(Cov.)')
    fig.colorbar(im, ax=sub) 
    plt.show() 
    return None 


def readPk(catalog, ell=0, sys=None): 
    ''' ***TESTED*** 
    test of reading in P(k). 
    '''
    # mocks
    pkay = Data.Pk() 
    n_mock = pkay._n_mock(catalog) 
    i_sample = np.random.choice(range(1,n_mock+1), 5, replace=False) 
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111) 

    for i in i_sample: 
        pkay.Read(catalog, i, ell=ell, sys=sys)
        k, pk = pkay.k, pkay.pk
        
        sub.plot(k, pk) 
    
    sub.set_xlim([1e-3, 0.5])
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_xscale('log') 
    sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
    sub.set_yscale('log') 

    plt.show() 
    return None


def Pk_rebin(catalog, rebin, ell=0, krange=None, sys=None): 
    ''' ***TESTED*** 
    Test the rebinning of P(k)  
    '''
    pkay = Data.Pk() 
    n_mock = pkay._n_mock(catalog) 
    i_sample = np.random.choice(range(1,n_mock+1), 5, replace=False) 
    
    prettyplot() 
    pretty_colors = prettycolors()
    fig = plt.figure() 
    sub = fig.add_subplot(111) 
    for ii, i in enumerate(i_sample): 
        offset = (ii+1)*2

        pkay.Read(catalog, i, ell=ell)
        k, pk = pkay.k, pkay.pk
        sub.plot(k, pk/offset, ls='--', c=pretty_colors[ii]) 
        print 'initially ', len(k), ' bins' 

        # impose krange and rebin 
        pkay.krange(krange)
        k, pk, cnt = pkay.rebin(rebin) 
        sub.scatter(k, pk/offset, c=pretty_colors[ii]) 
        print 'to ', len(k), ' bins' 

    sub.set_xlim([1e-3, 0.5])
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_xscale('log') 
    sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
    sub.set_yscale('log') 

    plt.show() 
    return None


def patchyPk_outlier(zbin, ell=0):
    ''' According to Florian there are 3 mocks with strange 
    P(k)s. Find them by examine P(k) that deviate significantly 
    from the mean. 
    '''
    catalog = 'patchy.ngc.z'+str(zbin)
    pkay = Data.Pk() 
    n_mock = pkay._n_mock(catalog) 

    prettyplot() 
    pretty_colors = prettycolors()
    fig = plt.figure() 
    sub = fig.add_subplot(111) 
    for i in range(1,n_mock+1):
        pkay.Read(catalog, i, ell=ell, sys='fc')
        k, pk = pkay.k, pkay.pk
        if i == 1: 
            pks = np.zeros((n_mock, len(k)))
        pks[i-1,:] = pk 

    mu_pk = np.sum(pks, axis=0)/np.float(n_mock)
    sig_pk = np.zeros(pks.shape[1])
    for ik in range(pks.shape[1]): 
        sig_pk[ik] = np.std(pks[:,ik]) 
    
    for i in range(1,n_mock+1):
        if ((pks[i-1,:] - mu_pk)/sig_pk).max() > 3.: 
            print i
            sub.plot(k, pks[i-1,:], lw=1) 
    sub.set_xlim([1e-3, 0.5])
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_xscale('log') 
    sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
    sub.set_yscale('log') 

    plt.show() 
    return None 


def Pk_i(catalog, i_mock, sys=None, rebin=None): 
    ''' test of reading in P(k). 
    '''
    # mocks
    pkay = Data.Pk() 
    n_mock = pkay._n_mock(catalog) 
    i_sample = np.random.choice(range(1,n_mock+1), 5, replace=False) 
    
    prettyplot() 
    fig = plt.figure(figsize=(21,8)) 
    for i_ell in range(3): 
        sub = fig.add_subplot(1,3,i_ell+1) 

        for i in i_sample: 
            pkay.Read(catalog, i, ell=2*i_ell, sys=sys)
            k, pk, _ = pkay.rebin(rebin)
            
            sub.plot(k, pk) 

        pkay.Read(catalog, i_mock, ell=2*i_ell, sys=sys)
        k, pk, _ = pkay.rebin(rebin)
        sub.plot(k, pk, lw=2, c='k') 
        
        sub.set_xlim([1e-3, 0.5])
        sub.set_xlabel('$\mathtt{k}$', fontsize=25)
        sub.set_xscale('log') 
        sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
        sub.set_yscale('log') 
    
    plt.show()
    return None


if __name__=="__main__":
    beutler_patchy_Cov_diag(1, NorS='ngc')
    beutler_patchy_Cov_diag(2, NorS='ngc')
    beutler_patchy_Cov_diag(3, NorS='ngc')
    #Data.boss_preprocess()

