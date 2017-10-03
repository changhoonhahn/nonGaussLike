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


def Beutler_BOSS_Plk(zbin): 
    ''' Compare the powerspectrum calculated using Roman's estimator 
    versus Florian's P(k)s
    '''
    if zbin == 1: 
        f_boss = ''.join([UT.catalog_dir('boss'), 'plk.galaxy_DR12v5_CMASSLOWZTOT_North.Lbox2800.Ngrid360.O4intp.P010000.fc.z1']) 
    elif zbin == 2: 
        f_boss = ''.join([UT.catalog_dir('boss'), 'plk.galaxy_DR12v5_CMASSLOWZTOT_North.Lbox3200.Ngrid410.O4intp.P010000.fc.z2']) 
    elif zbin == 3: 
        f_boss = ''.join([UT.catalog_dir('boss'), 'plk.galaxy_DR12v5_CMASSLOWZTOT_North.Lbox3800.Ngrid490.O4intp.P010000.fc.z3']) 

    k_boss, p0k, p2k, p4k, counts = np.loadtxt(f_boss, unpack=True, usecols=[0, 5, 2, 3, -2]) # k, p0(k), and number of modes 

    # read in Florian's P(k) 
    ks, pks = [], [] 
    for ell in [0, 2, 4]: 
        if ell == 0: 
            pole = 'monopole'
        elif ell == 2: 
            pole = 'quadrupole'
        elif ell == 4: 
            pole = 'hexadecapole'
        f_beut = ''.join([UT.dat_dir()+'Beutler/public_material_RSD/Beutleretal_pk_', pole, 
            '_DR12_NGC_z', str(zbin), '_prerecon_120.dat'])
        k_beut, plk = np.loadtxt(f_beut, skiprows=31, unpack=True, usecols=[1,2]) 
        pks.append(plk)
        ks.append(k_beut) 

    prettyplot()
    fig = plt.figure(figsize=(21, 8))
    sub = fig.add_subplot(131) # monopole comparison 
    sub.plot(k_boss, p0k) 
    sub.plot(ks[0], pks[0], label='Beutler+') 
    sub.set_xscale('log') 
    sub.set_yscale('log') 
    sub.set_xlim([0.01, 0.15]) 
    sub.set_ylim([5e3, 1.5e5]) 
    sub.set_xlabel('$\mathtt{k}$', fontsize=25) 
    sub.set_ylabel('$\mathtt{P_0(k)}$', fontsize=25) 
    sub.legend(loc='upper right') 

    sub = fig.add_subplot(132) # quadrupole comparison 
    sub.plot(k_boss, p2k) 
    sub.plot(ks[1], pks[1]) 

    sub.set_xscale('log') 
    sub.set_yscale('log') 
    sub.set_xlim([0.01, 0.15]) 
    sub.set_ylim([1e3, 5e4]) 
    sub.set_xlabel('$\mathtt{k}$', fontsize=25) 
    sub.set_ylabel('$\mathtt{P_2(k)}$', fontsize=25) 

    sub = fig.add_subplot(133) # quadrupole comparison 
    sub.plot(k_boss, p4k) 
    sub.plot(ks[2], pks[2]) 

    sub.set_xscale('log') 
    sub.set_yscale('log') 
    sub.set_xlim([0.01, 0.15]) 
    sub.set_xlabel('$\mathtt{k}$', fontsize=25) 
    sub.set_ylabel('$\mathtt{P_4(k)}$', fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'plk.boss.z', str(zbin), '.comparison.png']), bbox_inches='tight') 
    return None 


def beutler_patchy_Cov(zbin, ell=0, NorS='ngc'): 
    ''' compare my patchy covariance to Florian's 
    '''
    _,_,C_patchy = Data.patchyCov(zbin, NorS=NorS, ell=ell)
    _,_,C_beutler = Data.beutlerCov(zbin, NorS=NorS, ell=ell)
    
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


def beutler_patchy_Cov_diag(zbin, NorS='ngc', ell=0, clobber=False): 
    ''' compare the diagonal elements of my patchy covariance to Florian's 
    '''
    ki_pat,kj_pat,C_patchy = Data.patchyCov(zbin, NorS=NorS, ell=ell, clobber=clobber)
    ki_beu,kj_beu,C_beutler = Data.beutlerCov(zbin, NorS=NorS, ell=ell)

    prettyplot()
    fig = plt.figure(figsize=(10, 8))
    sub = fig.add_subplot(111)
    
    sub.plot(ki_pat, C_patchy.diagonal(), label='Patchy')
    sub.plot(ki_beu, C_beutler.diagonal(), c='k', ls='--', label='Beutler')
    sub.set_xscale('log') 
    sub.set_yscale('log') 
    sub.set_xlim([0.01, 0.15]) 
    #sub.set_ylim([5e3, 1.5e5]) 
    sub.set_xlabel('$\mathtt{k}$', fontsize=25) 
    sub.set_ylabel('$C_{i,i}$', fontsize=25) 
    #sub.legend(loc='upper right')
    fig.savefig(''.join([UT.fig_dir(), 'Cov_ii_beutler_patchy.z', str(zbin), '.', NorS, '.ell', str(ell), '.comparison.png']),
        bbox_inches='tight') 
    plt.close()
    return None 


def patchyCov(zbin, NorS='ngc'): 
    '''***TESTED*** 
    Test patchy covariance matrix calcuation 
    '''
    _,_,C_X = Data.patchyCov(zbin, NorS=NorS)
    
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
    _,_,C_X = Data.beutlerCov(zbin, NorS=NorS)
    
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
    for ell in [0, 2, 4]:
        beutler_patchy_Cov_diag(1, NorS='ngc', ell=ell, clobber=True)
