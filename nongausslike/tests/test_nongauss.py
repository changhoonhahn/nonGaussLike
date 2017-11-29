'''

Tests for each step of nongauss.py

'''
import time 
import numpy as np 
from scipy.stats import gaussian_kde as gkde
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity as skKDE 
from numpy.random import multivariate_normal as mvn 
from sklearn.mixture import GaussianMixture as GMix
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


def div_ICA(obv='pk.ngc', K=10, div_func='kl'):
    ''' compare the KL or Renyi divergence for different ICA decomposition algorithms 
    FastICA deflation, FastICA parallel, Infomax ICA 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    if obv == 'pk.ngc':  str_obv = 'P(k)'
    elif obv == 'gmf': str_obv = '\zeta(N)'
    lbls = [r'$D( '+str_obv+' \parallel \prod_i p_\mathrm{KDE}(P(k)_i^\mathrm{ICA}))$', 
            r'$D( '+str_obv+' \parallel \prod_i p_\mathrm{GMM}(P(k)_i^\mathrm{ICA}))$']

    icas = ['ICA', 'parICA'] 

    if obv == 'pk.ngc': 
        Nref = 2000
        hrange = [-0.5, 0.5]
    elif obv == 'gmf': 
        Nref = 10000
        hranges = [-0.1, 0.4]

    fig = plt.figure(figsize=(10,4))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_div, str_div in enumerate(['scottKDE.K'+str(K), 'GMM.K'+str(K)+'.ncomp30']): 

        divs = []
        for ica in icas: 
            f_div = ''.join([UT.dat_dir(), 'diverg.', obv, 
                '.pXi_', ica, '_', str_div, '.Nref', str(Nref), '.', div_func, '.dat']) 
            try: 
                div = np.loadtxt(f_div)
            except IOError: 
                print f_div
                continue 
            divs.append(div) 

        nbins = 50
        sub = fig.add_subplot(1,2,i_div+1)
        y_max = 0. 
        for div, ica in zip(divs, icas): 
            print np.mean(div)
            hh = np.histogram(div, normed=True, range=hrange, bins=nbins)
            bp = UT.bar_plot(*hh) 
            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', label=ica) 
            y_max = max(y_max, bp[1].max()) 
        if i_div == 0: sub.legend(loc='upper left', prop={'size': 20}) 
        sub.set_xlim(hrange)  
        sub.set_ylim([0., y_max*1.4]) 
        sub.set_title(lbls[i_div]) 

    if div_func == 'kl': 
        bkgd.set_xlabel(r'KL divergence', fontsize=20, labelpad=20)
    elif div_func == 'renyi0.5': 
        bkgd.set_xlabel(r'R\'enyi-$\alpha$ divergence', fontsize=20, labelpad=20)
    bkgd.set_xticklabels([])
    bkgd.set_yticklabels([])
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    fig.subplots_adjust(wspace=.15, hspace=0.3)
    f_fig = ''.join([UT.fig_dir(), 'tests/',
        'ICA_kNNdiverg.', obv, '.K', str(K), '.', div_func, '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def div_K(div_func='kl'):
    ''' compare the KL or Renyi divergence for the following with their using different K values 
    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    lbls = [r'$D( P(k) \parallel \mathcal{N}({\bf C}))$',
            r'$D( P(k) \parallel p_\mathrm{KDE}(P(k)))$',
            r'$D( P(k) \parallel p_\mathrm{GMM}(P(k)))$',
            r'$D( P(k) \parallel \prod_i p_\mathrm{KDE}(P(k)_i^\mathrm{ICA}))$', 
            r'$D( P(k) \parallel \prod_i p_\mathrm{GMM}(P(k)_i^\mathrm{ICA}))$']

    fig = plt.figure(figsize=(20,4))
    for i_obv, obv in enumerate(['pk.ngc', 'gmf']):
        if obv == 'pk.ngc': 
            Nref = 2000
            if div_func == 'kl': hranges = [[-0.5, 0.5], [-0.5, 7.], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]##7.]
            else: hranges = [[-0.5, 0.5] for i in range(5)]
            Ks = [5, 10, 15] 
        elif obv == 'gmf': 
            Nref = 10000
            hranges = [[-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4]]##7.]
            Ks = [10] 

        for K in Ks: 
            fs = ['pX_gauss.K'+str(K), 'pX_scottKDE.K'+str(K), 'pX_GMM.K'+str(K)+'.ncomp30', 
                'pXi_ICA_scottKDE.K'+str(K), 'pXi_ICA_GMM.K'+str(K)+'.ncomp30'] 
            divs, divs_ref = [], [] 
            for f in fs: 
                f_div = ''.join([UT.dat_dir(), 'diverg.', obv, '.', f, '.Nref', str(Nref), '.', 
                    div_func, '.dat']) 
                try: 
                    div = np.loadtxt(f_div)
                except IOError: 
                    print f_div
                    continue 
                divs.append(div) 
         
            nbins = 50
            bkgd = fig.add_subplot(2,1,i_obv+1, frameon=False)
            for i_div, div, lbl in zip(range(len(fs)), divs, lbls): 
                sub = fig.add_subplot(2,5,len(fs)*i_obv+i_div+1)
                y_max = 0. 
                hh = np.histogram(div, normed=True, range=hranges[i_div], bins=nbins)
                bp = UT.bar_plot(*hh) 
                sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
                y_max = max(y_max, bp[1].max()) 
                sub.set_xlim(hranges[i_div])  
                sub.set_ylim([0., y_max*1.4]) 
                if i_obv == 0: 
                    sub.set_title(lbl) 
    
        if div_func == 'kl': 
            bkgd.set_xlabel(r'KL divergence', fontsize=20, labelpad=20)
        elif div_func == 'renyi0.5': 
            bkgd.set_xlabel(r'R\'enyi-$\alpha$ divergence', fontsize=20, labelpad=20)
        bkgd.set_xticklabels([])
        bkgd.set_yticklabels([])
        bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    fig.subplots_adjust(wspace=.15, hspace=0.3)
    f_fig = ''.join([UT.fig_dir(), 'tests/Ktest_kNNdiverg.', div_func, '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def delta_div(K=10, div_func='kl'):
    ''' compare the KL divergence for the following with their
    reference divergence counterpart

    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    f_refs = ['ref.K'+str(K), 'pX_scottKDE_ref.K'+str(K), 'pX_GMM_ref.K'+str(K)+'.ncomp30', 
            'pXi_parICA_scottKDE_ref.K'+str(K), 'pXi_parICA_GMM_ref.K'+str(K)+'.ncomp30']
    fs = ['pX_gauss.K'+str(K), 'pX_scottKDE.K'+str(K), 'pX_GMM.K'+str(K)+'.ncomp30', 
            'pXi_parICA_scottKDE.K'+str(K), 'pXi_ICA_GMM.K'+str(K)+'.ncomp30'] 
    lbls = [r'$D( P(k) \parallel \mathcal{N}({\bf C}))$',
            r'$D( P(k) \parallel p_\mathrm{KDE}(P(k)))$',
            r'$D( P(k) \parallel p_\mathrm{GMM}(P(k)))$',
            r'$D( P(k) \parallel \prod_i p_\mathrm{KDE}(P(k)_i^\mathrm{ICA}))$', 
            r'$D( P(k) \parallel \prod_i p_\mathrm{GMM}(P(k)_i^\mathrm{ICA}))$']

    fig = plt.figure(figsize=(20,4))
    for i_obv, obv in enumerate(['pk.ngc', 'gmf']):
        if obv == 'pk.ngc': 
            Nref = 2000
            if div_func == 'kl': 
                hranges = [[-0.5, 0.5], [-0.5, 7.], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]##7.]
            else: 
                hranges = [[-0.5, 0.5] for i in range(5)]
        elif obv == 'gmf': 
            Nref = 10000
            hranges = [[-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4], [-0.1, 0.4]]##7.]

        divs, divs_ref = [], [] 
        for f, f_ref in zip(fs, f_refs): 
            f_div = ''.join([UT.dat_dir(), 'diverg.', obv, '.', f, '.Nref', str(Nref), '.', 
                div_func, '.dat']) 
            f_div_ref = ''.join([UT.dat_dir(), 'diverg.', obv, '.', f_ref, '.Nref', str(Nref), 
                '.', div_func, '.dat']) 
            try: 
                div = np.loadtxt(f_div)
                div_ref = np.loadtxt(f_div_ref)
            except IOError: 
                print f_div
                print f_div_ref
                continue 
            divs.append(div) 
            divs_ref.append(div_ref) 
     
        nbins = 50
        bkgd = fig.add_subplot(2,1,i_obv+1, frameon=False)
        for i_div, div, div_ref, lbl in zip(range(len(fs)), divs, divs_ref, lbls): 
            sub = fig.add_subplot(2,5,len(fs)*i_obv+i_div+1)

            y_max = 0. 
            hh = np.histogram(div_ref, normed=True, range=hranges[i_div], bins=nbins)
            bp = UT.bar_plot(*hh) 
            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
            y_max = max(y_max, bp[1].max()) 

            hh = np.histogram(div, normed=True, range=hranges[i_div], bins=nbins)
            bp = UT.bar_plot(*hh) 
            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
            y_max = max(y_max, bp[1].max()) 

            sub.text(0.8, 0.9, str(round(np.mean(div) - np.mean(div_ref),2)), ha='left', va='top', 
                    transform=sub.transAxes, fontsize=15)

            sub.set_xlim(hranges[i_div])  
            sub.set_ylim([0., y_max*1.4]) 
            if i_obv == 0: 
                sub.set_title(lbl) 
    
        if div_func == 'kl': 
            bkgd.set_xlabel(r'KL divergence', fontsize=20, labelpad=20)
        elif div_func == 'renyi0.5': 
            bkgd.set_xlabel(r'R\'enyi-$\alpha$ divergence', fontsize=20, labelpad=20)
        bkgd.set_xticklabels([])
        bkgd.set_yticklabels([])
        bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    fig.subplots_adjust(wspace=.15, hspace=0.3)
    f_fig = ''.join([UT.fig_dir(), 'tests/delta_kNNdiverg.K', str(K), '.', div_func, '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def diverge(obvs, div_func='kl', Nref=1000, K=5, n_mc=10, n_comp_max=10, n_mocks=2000, 
        pk_mock='patchy.z1', NorS='ngc'):
    ''' compare the divergence estimates between 
    D( gauss(C_X) || gauss(C_X) ),  D( mock X || gauss(C_X)), 
    D( mock X || p(X) KDE), D( mock X || p(X) GMM), 
    D( mock X || PI p(X^i_ICA) KDE), and D( mock X || PI p(X^i_ICA) GMM)
    '''
    if isinstance(Nref, float): Nref = int(Nref)
    # read in mock data X  
    if obvs == 'pk': 
        X_mock = NG.X_pk_all(pk_mock, NorS=NorS, sys='fc')[:n_mocks]
    elif obvs == 'gmf': 
        X_mock = NG.X_gmf_all()[:n_mocks]
    else: 
        raise ValueError("obvs = 'pk' or 'gmf'")  
    n_mock = X_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    X_mock_meansub, _ = NG.meansub(X_mock) # mean subtract
    X_w, W = NG.whiten(X_mock_meansub)
    X_ica, _ = NG.Ica(X_w)  # ICA transformation 
    C_X = np.cov(X_w.T) # covariance matrix

    # p(mock X) GMM
    gmms, bics = [], [] 
    for i_comp in range(1,n_comp_max+1):
        gmm = GMix(n_components=i_comp)
        gmm.fit(X_w) 
        gmms.append(gmm)
        bics.append(gmm.bic(X_w))
    ibest = np.array(bics).argmin() 
    kern_gmm = gmms[ibest]

    # p(mock X) KDE 
    t0 = time.time() 
    grid = GridSearchCV(skKDE(),
            {'bandwidth': np.linspace(0.1, 1.0, 30)},
            cv=10) # 10-fold cross-validation
    grid.fit(X_w)
    kern_kde = grid.best_estimator_
    dt = time.time() - t0 
    print('%f sec' % dt) 
    
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

    # caluclate the divergences now 
    div_gauss_ref, div_gauss = [], []
    div_gmm, div_gmm_ica = [], [] 
    div_kde, div_kde_ica = [], [] 
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        t_start = time.time() 
        # reference divergence in order to showcase the estimator's scatter
        # Gaussian distribution described by C_gmf with same n_mock mocks 
        gauss = mvn(np.zeros(X_mock.shape[1]), C_X, size=n_mock)
        div_gauss_ref_i = NG.kNNdiv_gauss(gauss, C_X, Knn=K, div_func=div_func, Nref=Nref)
        div_gauss_ref.append(div_gauss_ref_i)
        # estimate divergence between gmfs_white and a 
        # Gaussian distribution described by C_gmf
        div_gauss_i = NG.kNNdiv_gauss(X_w, C_X, Knn=K, div_func=div_func, Nref=Nref)
        div_gauss.append(div_gauss_i)
        # D( mock X || p(X) GMM)
        div_gmm_i = NG.kNNdiv_Kernel(X_w, kern_gmm, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=False) 
        div_gmm.append(div_gmm_i)
        # D( mock X || p(X) KDE)
        div_kde_i = NG.kNNdiv_Kernel(X_w, kern_kde, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=False) 
        div_kde.append(div_kde_i)
        # D( mock X || PI p(X^i_ICA) GMM), 
        div_gmm_ica_i = NG.kNNdiv_Kernel(X_ica, kern_gmm_ica, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=True)
        div_gmm_ica.append(div_gmm_ica_i)
        # D( mock X || PI p(X^i_ICA) KDE), 
        div_kde_ica_i = NG.kNNdiv_Kernel(X_ica, kern_kde_ica, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=True)
        div_kde_ica.append(div_kde_ica_i)
        print('t= %f sec' % round(time.time()-t_start,2))
    
    divs = [div_gauss_ref, div_gauss, div_gmm, div_kde, div_gmm_ica, div_kde_ica]
    labels = ['Ref.', r'$D(\{\zeta_i^{(m)}\}\parallel \mathcal{N}({\bf C}^{(m)}))$', 
            r'$D(\{\zeta^{(m)}\}\parallel p_\mathrm{GMM}(\{\zeta^{m}\}))$',
            r'$D(\{\zeta^{(m)}\}\parallel p_\mathrm{KDE}(\{\zeta^{m}\}))$',
            r'$D(\{\zeta_\mathrm{ICA}^{(m)}\}\parallel \prod_{i} p^\mathrm{GMM}(\{\zeta_{i, \mathrm{ICA}}^{m}\}))$', 
            r'$D(\{\zeta_\mathrm{ICA}^{(m)}\}\parallel \prod_{i} p^\mathrm{KDE}(\{\zeta_{i, \mathrm{ICA}}^{m}\}))$']

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    if obvs == 'pk': 
        x_min, x_max = 0., 0.
        for div in divs: 
            x_min = min(x_min, np.min(div)) 
            x_max = max(x_max, np.max(div)) 
        hrange = [x_min, x_max]
    elif obvs == 'gmf':
        hrange = [-0.15, 0.6]

    nbins = 50
    y_max = 0.
    for div, lbl in zip(divs, labels): 
        hh = np.histogram(np.array(div), normed=True, range=hrange, bins=nbins)
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', 
                alpha=0.5, label=lbl) 
        y_max = max(y_max, bp[1].max()) 
        if (np.average(div) < hrange[0]) or (np.average(div) > hrange[1]): 
            print('divergence of %s (%f) is outside range' % (lbl, np.average(div)))
    sub.set_xlim(hrange) 
    sub.set_ylim([0., y_max*1.2]) 
    sub.legend(loc='upper left', prop={'size': 15})
    # xlabels
    if 'renyi' in div_func: 
        alpha = float(div_func.split(':')[-1])
        sub.set_xlabel(r'Renyi-$\alpha='+str(alpha)+'$ divergence', fontsize=20)
    elif 'kl' in div_func: 
        sub.set_xlabel(r'KL divergence', fontsize=20)
    if 'renyi' in div_func: str_div = 'renyi'+str(alpha) 
    elif div_func == 'kl': str_div = 'kl'
    f_fig = ''.join([UT.fig_dir(), 'tests/kNN_divergence.', obvs, '.K', str(K), 
        '.', str(n_mocks), '.', str_div, '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def divGMF(div_func='kl', Nref=1000, K=5, n_mc=10, n_comp_max=10, n_mocks=2000):
    ''' compare the divergence estimates between 
    D( gauss(C_gmf) || gauss(C_gmf) ),  D( gmfs || gauss(C_gmf) ), 
    D( gmfs || p(gmfs) KDE), D( gmfs || p(gmfs) GMM), 
    D( gmfs || PI p(gmfs^i_ICA) KDE), and D( gmfs || PI p(gmfs^i_ICA) GMM)
    '''
    if isinstance(Nref, float): 
        Nref = int(Nref)
    # read in mock GMFs from all HOD realizations (20,000 mocks)
    gmfs_mock = NG.X_gmf_all()[:n_mocks]
    n_mock = gmfs_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    gmfs_mock_meansub, _ = NG.meansub(gmfs_mock) # mean subtract
    X_w, W = NG.whiten(gmfs_mock_meansub)
    X_ica, _ = NG.Ica(X_w)  # ICA transformation 

    C_gmf = np.cov(X_w.T) # covariance matrix

    # p(gmfs) GMM
    gmms, bics = [], [] 
    for i_comp in range(1,n_comp_max+1):
        gmm = GMix(n_components=i_comp)
        gmm.fit(X_w) 
        gmms.append(gmm)
        bics.append(gmm.bic(X_w))
    ibest = np.array(bics).argmin() 
    kern_gmm = gmms[ibest]

    # p(gmfs) KDE 
    t0 = time.time() 
    grid = GridSearchCV(skKDE(),
            {'bandwidth': np.linspace(0.1, 1.0, 30)},
            cv=10) # 10-fold cross-validation
    grid.fit(X_w)
    kern_kde = grid.best_estimator_
    dt = time.time() - t0 
    print('%f sec' % dt) 
    
    # PI p(gmfs^i_ICA) GMM
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
    
    # PI p(gmfs^i_ICA) KDE  
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

    # caluclate the divergences now 
    div_gauss_ref, div_gauss = [], []
    div_gmm, div_gmm_ica = [], [] 
    div_kde, div_kde_ica = [], [] 
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        t_start = time.time() 
        # reference divergence in order to showcase the estimator's scatter
        # Gaussian distribution described by C_gmf with same n_mock mocks 
        gauss = mvn(np.zeros(gmfs_mock.shape[1]), C_gmf, size=n_mock)
        div_gauss_ref_i = NG.kNNdiv_gauss(gauss, C_gmf, Knn=K, div_func=div_func, Nref=Nref)
        div_gauss_ref.append(div_gauss_ref_i)
        # estimate divergence between gmfs_white and a 
        # Gaussian distribution described by C_gmf
        div_gauss_i = NG.kNNdiv_gauss(X_w, C_gmf, Knn=K, div_func=div_func, Nref=Nref)
        div_gauss.append(div_gauss_i)
        # D( gmfs || p(gmfs) GMM)
        div_gmm_i = NG.kNNdiv_Kernel(X_w, kern_gmm, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=False) 
        div_gmm.append(div_gmm_i)
        # D( gmfs || p(gmfs) KDE)
        div_kde_i = NG.kNNdiv_Kernel(X_w, kern_kde, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=False) 
        div_kde.append(div_kde_i)
        # D( gmfs || PI p(gmfs^i_ICA) GMM), 
        div_gmm_ica_i = NG.kNNdiv_Kernel(X_ica, kern_gmm_ica, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=True)
        div_gmm_ica.append(div_gmm_ica_i)
        # D( gmfs || PI p(gmfs^i_ICA) KDE), 
        div_kde_ica_i = NG.kNNdiv_Kernel(X_ica, kern_kde_ica, Knn=K, div_func=div_func, 
                Nref=Nref, compwise=True)
        div_kde_ica.append(div_kde_ica_i)
        print('t= %f sec' % round(time.time()-t_start,2))

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    hrange = [-0.15, 0.6]
    nbins = 50
    
    divs = [div_gauss_ref, div_gauss, div_gmm, div_kde, div_gmm_ica, div_kde_ica]
    labels = ['Ref.', r'$D(\{\zeta_i^{(m)}\}\parallel \mathcal{N}({\bf C}^{(m)}))$', 
            r'$D(\{\zeta^{(m)}\}\parallel p_\mathrm{GMM}(\{\zeta^{m}\}))$',
            r'$D(\{\zeta^{(m)}\}\parallel p_\mathrm{KDE}(\{\zeta^{m}\}))$',
            r'$D(\{\zeta_\mathrm{ICA}^{(m)}\}\parallel \prod_{i} p^\mathrm{GMM}(\{\zeta_{i, \mathrm{ICA}}^{m}\}))$', 
            r'$D(\{\zeta_\mathrm{ICA}^{(m)}\}\parallel \prod_{i} p^\mathrm{KDE}(\{\zeta_{i, \mathrm{ICA}}^{m}\}))$']
    y_max = 0.
    for div, lbl in zip(divs, labels): 
        hh = np.histogram(np.array(div), normed=True, range=hrange, bins=nbins)
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', 
                alpha=0.5, label=lbl) 
        y_max = max(y_max, bp[1].max()) 
        if (np.average(div) < hrange[0]) or (np.average(div) > hrange[1]): 
            print('divergence of %s (%f) is outside range' % (lbl, np.average(div)))
    sub.set_xlim(hrange) 
    sub.set_ylim([0., y_max*1.2]) 
    sub.legend(loc='upper left', prop={'size': 15})
    # xlabels
    if 'renyi' in div_func: 
        alpha = float(div_func.split(':')[-1])
        sub.set_xlabel(r'Renyi-$\alpha='+str(alpha)+'$ divergence', fontsize=20)
    elif 'kl' in div_func: 
        sub.set_xlabel(r'KL divergence', fontsize=20)
    if 'renyi' in div_func: str_div = 'renyi'+str(alpha) 
    elif div_func == 'kl': str_div = 'kl'
    f_fig = ''.join([UT.fig_dir(), 'tests/kNN_divergence.gmf.K', str(K), '.', str(n_mocks), 
        '.', str_div, '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def GMF_p_Xw_i(ica=False, pca=False): 
    ''' Test the probability distribution function of each transformed X
    component -- p(X^i). First compare the histograms of p(X_w^i) 
    with N(0,1). Then compare the gaussian KDE of p(X_w^i).
    '''
    gmf = NG.X_gmf_all() # import all the GMF mocks 
    X, _ = NG.meansub(gmf)
    str_w = 'W'
    if ica and pca: raise ValueError
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
    fig = plt.figure(figsize=(5*gmf.shape[1],4))
    for icomp in range(gmf.shape[1]): 
        sub = fig.add_subplot(1, gmf.shape[1], icomp+1)
        # histogram of X_w^i s 
        hh = np.histogram(X_w[:,icomp], normed=True, bins=50, range=[-5., 5.])
        p_X_w_arr = UT.bar_plot(*hh)
        sub.fill_between(p_X_w_arr[0], np.zeros(len(p_X_w_arr[1])), p_X_w_arr[1], 
                color='k', alpha=0.25)
        x = np.linspace(-5., 5., 100)
        sub.plot(x, UT.gauss(x, 1., 0.), c='k', lw=2, ls=':', label='$\mathcal{N}(0,1)$')
        # p(X_w^i) gaussian KDE fits  
        t_start = time.time() 
        pdf = NG.p_Xw_i(X_w, icomp, x=x, method='gkde')
        sub.plot(x, pdf, lw=2.5, label='Gaussian KDE')
        print 'scipy Gaussian KDE ', time.time()-t_start
        # p(X_w^i) SKlearn KDE fits  
        t_start = time.time() 
        pdf = NG.p_Xw_i(X_w, icomp, x=x, method='sk_kde')
        sub.plot(x, pdf, lw=2.5, label='SKlearn KDE')
        print 'SKlearn CV best-fit KDE ', time.time()-t_start
        # p(X_w^i) statsmodels KDE fits  
        t_start = time.time() 
        pdf = NG.p_Xw_i(X_w, icomp, x=x, method='sm_kde')
        sub.plot(x, pdf, lw=2.5, label='StatsModels KDE')
        print 'Stats Models KDE ', time.time()-t_start
        # p(X_w^i) GMM fits  
        pdf = NG.p_Xw_i(X_w, icomp, x=x, method='gmm', n_comp_max=20)
        sub.plot(x, pdf, lw=2.5, ls='--', label='GMM')
        sub.set_xlim([-3., 3.])
        sub.set_xlabel('$X_{'+str_w+'}^{('+str(icomp)+')}$', fontsize=25) 
        sub.set_ylim([0., 0.6])
        if icomp == 0: 
            sub.set_ylabel('$P(X_{'+str_w+'})$', fontsize=25) 
            sub.legend(loc='upper left', prop={'size': 15}) 

    str_ica, str_pca = '', ''
    if ica: str_ica = '.ICA'
    if pca: str_pca = '.PCA'

    f = ''.join([UT.fig_dir(), 'tests/test.GMF_p_Xw_i', str_pca, str_ica, '.png'])
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


def X_gmf_all():
    ''' ***TESTED -- Nov 8, 2017***
    Test to make sure that NG.X_gmf_all returns correct values
    '''
    X, nbins = NG.X_gmf_all(n_arr=True) 
    nmid = 0.5*(nbins[1:] + nbins[:-1])
    assert X.shape[1] == len(nmid)
    assert X.shape[0] == 20000

    fig = plt.figure(figsize=(5,5)) 
    sub = fig.add_subplot(111)
    for i in np.random.choice(range(X.shape[0]), 1000, replace=False):
        sub.plot(nmid, X[i,:], c='k', lw=0.01)
    sub.plot(nmid, np.average(X, axis=0), c='r', lw=2, ls='--')
    # x-axis
    sub.set_xlim([0., 180.]) 
    sub.set_xlabel('$N$', fontsize=20) 
    # y-axis
    sub.set_yscale('log') 
    sub.set_ylabel(r'$\zeta(N)$', fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'tests/X_gmf_all.png']), bbox_inches='tight') 
    plt.close() 
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
    #for k in [5,10,15]: 
    #    div_ICA(obv='pk.ngc', K=k, div_func='k')
    #    div_ICA(obv='pk.ngc', K=k, div_func='renyi0.5')
    div_K(div_func='renyi0.5')
    div_K(div_func='kl')
    #delta_div(K=10, div_func='kl')
    #delta_div(K=10, div_func='renyi0.5')
    #diverge('pk', div_func='kl', Nref=5000, K=10, n_mc=10, n_comp_max=10, n_mocks=2000, 
    #    pk_mock='patchy.z1', NorS='ngc')

    #for n in [2000, 4000, 6000]: 
    #    divGMF(n_mocks=n, div_func='kl', Nref=n*1.5, K=10, n_mc=100, n_comp_max=20)
    #GMF_p_Xw_i(ica=True, pca=False)
    #lnL_pca_gauss('patchy.z1', ell=0, krange=[0.01, 0.15], NorS='ngc')
