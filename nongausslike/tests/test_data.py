'''
'''
import numpy as np 

# --- local ---
import env
import data as Data

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


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
    P(k)s
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
        sub.plot(k, pk, c='k') 
    sub.set_xlim([1e-3, 0.5])
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_xscale('log') 
    sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
    sub.set_yscale('log') 

    plt.show() 
    return None 


if __name__=="__main__":
    patchyPk_outlier(1, ell=0)
