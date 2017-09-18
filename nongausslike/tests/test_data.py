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


def readPk(catalog, ell=0): 
    ''' ***TESTED*** 
    test of reading in P(k). 
    '''
    # mocks
    if catalog == 'nseries': 
        i_sample = np.random.choice(range(1,85), 5, replace=False) 
    elif catalog == 'qpm':   
        i_sample = np.random.choice(range(1,1001), 5, replace=False) 

    pkay = Data.Pk() 
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111) 

    for i in i_sample: 
        pkay.Read(catalog, i, ell=ell)
        k, pk = pkay.k, pkay.pk
        
        sub.plot(k, pk) 
    
    sub.set_xlim([1e-3, 0.5])
    sub.set_xlabel('$\mathtt{k}$', fontsize=25)
    sub.set_xscale('log') 
    sub.set_ylabel('$\mathtt{P(k)}$', fontsize=25)
    sub.set_yscale('log') 

    plt.show() 
    return None


def Pk_rebin(catalog, rebin, ell=0, krange=None): 
    ''' ***TESTED*** 
    Test the rebinning of P(k)  
    '''
    # mocks
    if catalog == 'nseries': 
        i_sample = np.random.choice(range(1,85), 5, replace=False) 
    elif catalog == 'qpm':   
        i_sample = np.random.choice(range(1,1001), 5, replace=False) 

    pkay = Data.Pk() 
    
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


if __name__=="__main__":
    Pk_rebin('qpm', 'beutler', ell=2, krange=[0.01, 0.15])
