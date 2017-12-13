'''

plots for the paper  


'''
import time 
import numpy as np 
import corner as DFM
import wquantiles as wq
from statsmodels.nonparametric import kde 
from scipy.stats import gaussian_kde as gkde
from sklearn.neighbors import KernelDensity as skKDE 
from numpy.random import multivariate_normal as mvn 
from sklearn.mixture import GaussianMixture as GMix
from scipy.stats import multivariate_normal as mGauss
from sklearn.model_selection import GridSearchCV

# kNN-Divergence
from skl_groups.features import Features
from skl_groups.divergences import KNNDivergenceEstimator

# -- local -- 
import data as Data
import util as UT 
import infer as Inf
import nongauss as NG 

# -- plotting -- 
from ChangTools.plotting import prettycolors
from matplotlib import lines as mlines
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


def _div_Gauss_gmf(K=10):
    ''' compare the renyi-alpha and KL divergence for the following  
    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    for only GMF 
    '''
    # D( gauss(C_X) || gauss(C_X) ) vs D( mock X || gauss(C_X))
    fs = ['ref.K'+str(K), 'pX_gauss.K'+str(K)] 
    lbls = [ r'$D\left( \{x_i \in \mathcal{N}(\bar{\zeta}, \mathcal{C})\}\vert_{i=1}^{N_\mathrm{mock}} \parallel \mathcal{N}(\bar{\zeta}, \mathcal{C})\right)$', 
            r'$D\left(\{ \zeta_i^\mathrm{mock} \}\vert_{i=1}^{N_\mathrm{mock}} \parallel \mathcal{N}(\bar{\zeta}, \mathcal{C})\right)$']
    
    fig = plt.figure(figsize=(12,4))
    obvs = 'gmf'
    for i_div, div_func in enumerate(['renyi0.5', 'kl']): 
        divs = []  
        for f in fs: 
            if obvs == 'pk': 
                f_div = ''.join([UT.dat_dir(), 
                    'diverg.pk.ngc.', f, '.Nref2000.', div_func, '.dat']) 
            elif obvs == 'gmf': 
                f_div = ''.join([UT.dat_dir(), 
                    'diverg.gmf.', f, '.Nref10000.', div_func, '.dat']) 
            div = np.loadtxt(f_div)
            divs.append(div) 

        if i_div == 0: _lbls = [lbls[0], None]
        else: _lbls = [None, lbls[1]]
        
        hrange = [-0.1, 0.25]
        nbins = 40
        y_max = 0. 
        sub = fig.add_subplot(1,2,i_div+1)
        for div, lbl in zip(divs, _lbls): 
            hh = np.histogram(div, normed=True, range=hrange, bins=nbins)
            bp = UT.bar_plot(*hh) 
            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', label=lbl) 
            y_max = max(y_max, bp[1].max()) 
        sub.set_xlim(hrange) 
        sub.set_ylim([0., y_max*1.4]) 
        sub.legend(loc='upper right', prop={'size': 15})
        if div_func == 'kl': sub.set_xlabel(r'KL divergence', fontsize=20)
        elif div_func == 'renyi0.5': 
            sub.set_xlabel(r'R\'enyi-$\alpha$ divergence', fontsize=20)
    fig.subplots_adjust(wspace=.15)
    f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNNdiverg_Gauss_gmf.pdf'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def div_Gauss(K=10):
    ''' compare the renyi-alpha and KL divergence for the following  
    - D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    for both P(k) and GMF 
    '''
    # D( gauss(C_X) || gauss(C_X) ) vs D( mock X || gauss(C_X))
    fs = ['ref.K'+str(K), 'pX_gauss.K'+str(K)] 
    lbls = ['Ref.', r'$D(\textbf{X}^\mathrm{mock} \parallel \textbf{Y}^\mathrm{ref})$'] 
    
    fig = plt.figure(figsize=(12,5))
    for i_obv, obvs in enumerate(['pk', 'gmf']): 
        for i_div, div_func in enumerate(['renyi0.5', 'kl']): 
            divs = []  
            for f in fs: 
                if obvs == 'pk': 
                    f_div = ''.join([UT.dat_dir(), 
                        'diverg.pk.ngc.', f, '.Nref2000.', div_func, '.dat']) 
                elif obvs == 'gmf': 
                    f_div = ''.join([UT.dat_dir(), 
                        'diverg.gmf.', f, '.Nref10000.', div_func, '.dat']) 
                div = np.loadtxt(f_div)
                divs.append(div) 
            
            if obvs == 'pk': hrange = [-0.5, 0.5]
            elif obvs == 'gmf': hrange = [-0.1, 0.25]
            nbins = 40
            y_max = 0. 
            sub = fig.add_subplot(2,2,2*i_obv+i_div+1)
            for div, lbl in zip(divs, lbls): 
                hh = np.histogram(div, normed=True, range=hrange, bins=nbins)
                bp = UT.bar_plot(*hh) 
                sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', label=lbl) 
                y_max = max(y_max, bp[1].max()) 
            sub.set_xlim(hrange) 
            sub.set_ylim([0., y_max*1.4]) 
            if i_obv == 0: 
                if i_div == 1: sub.legend(loc='upper right', prop={'size': 15})
                else: 
                    sub.text(0.075, 0.85, r'$P_\ell(k)$', ha='left', va='top', 
                        transform=sub.transAxes, fontsize=20)
            elif i_obv == 1: 
                if div_func == 'kl': sub.set_xlabel(r'KL divergence', fontsize=20)
                elif div_func == 'renyi0.5': 
                    sub.set_xlabel(r'R\'enyi-$\alpha$ divergence', 
                        fontsize=20)
                    sub.text(0.075, 0.85, r'$\zeta(N)$', ha='left', va='top', 
                            transform=sub.transAxes, fontsize=20)
    fig.subplots_adjust(wspace=.15)
    f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNNdiverg_Gauss.pdf'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def div_nonGauss(K=10):
    ''' compare the renyi-alpha and KL divergence for the following  

    - reference D( gauss(C_X) || gauss(C_X) ) 
    - D( mock X || gauss(C_X))
    - D( mock X || p(X) GMM) 
    '''
    pretty_colors = prettycolors() 
    fig = plt.figure(figsize=(12,5))
    for i_obv, obv in enumerate(['pk.ngc', 'gmf']): 
        if obv == 'pk.ngc': Nref, ncomp, hrange = 2000, 30, [-0.5, 0.4] 
        elif obv == 'gmf': Nref, ncomp, hrange = 10000, 30, [-0.1, 0.2]
        lbls = [None, None, 
                r'$D( \textbf{X}^\mathrm{mock} \parallel \sim p_\mathrm{GMM}(\textbf{X}^\mathrm{mock}))$', 
                r'$D( \textbf{X}^\mathrm{mock} \parallel \sim \prod p_\mathrm{KDE}(\textbf{X}_i^\mathrm{ICA}))$']

        fs = ['ref.K'+str(K), 'pX_gauss.K'+str(K), 'pX_GMM.K'+str(K)+'.ncomp'+str(ncomp), 'pXi_parICA_scottKDE.K'+str(K)]

        for i_div, div_func in enumerate(['renyi0.5', 'kl']): 
            divs = []  
            for f in fs: 
                f_div = ''.join([UT.dat_dir(), 'diverg.', obv, '.', f, '.Nref', str(Nref), '.', div_func, '.dat']) 
                try: 
                    div = np.loadtxt(f_div)
                except IOError: 
                    print f_div
                    continue 
                divs.append(div) 
         
            nbins = [50, 100]
            y_max = 0. 
            sub = fig.add_subplot(2,2,2*i_obv+i_div+1)
            for ii, div, lbl in zip(range(len(divs)), divs, lbls): 
                hh = np.histogram(div, normed=True, range=hrange, bins=nbins[i_obv])
                bp = UT.bar_plot(*hh) 
                if ii == 2: 
                    _lbls = [lbl, None]
                    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], 
                            facecolor='none', hatch='//', edgecolor=pretty_colors[0], label=_lbls[i_obv]) 
                elif ii == 3: 
                    _lbls = [None, lbl]
                    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], 
                            facecolor=pretty_colors[5], alpha=0.75, label=_lbls[i_obv]) 
                else: 
                    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
                y_max = max(y_max, bp[1].max()) 
            sub.set_xlim(hrange) 
            sub.set_ylim([0., y_max*1.4]) 

            if i_obv == 0: 
                if i_div == 1: sub.legend(loc='upper right', ncol=2, prop={'size': 15})
                else: 
                    sub.text(0.075, 0.85, r'$P_\ell(k)$', ha='left', va='top', 
                        transform=sub.transAxes, fontsize=20)
            elif i_obv == 1: 
                if div_func == 'kl': 
                    sub.set_xlabel(r'KL divergence', fontsize=20)
                    sub.legend(loc='upper right', prop={'size': 15})
                elif div_func == 'renyi0.5': 
                    sub.set_xlabel(r'R\'enyi-$\alpha$ divergence', 
                        fontsize=20)
                    sub.text(0.075, 0.85, r'$\zeta(N)$', ha='left', va='top', 
                            transform=sub.transAxes, fontsize=20)
    fig.subplots_adjust(wspace=.15, hspace=.2)
    f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNNdiverg_nonGauss.pdf'])
    fig.savefig(f_fig, bbox_inches='tight') 
    return None


def GMM_pedagog(): 
    ''' A pedagogical demonstration of how GMM works.
    '''
    ## read in X_mock from GMF data
    #X_mock = NG.X_gmf_all()[:,-1]
    ## mean subtract
    #X_mock_meansub, _ = NG.meansub(X_mock) 
    ## and normalize by sigma 
    #sig = np.std(X_mock_meansub) 
    #X_mock_meansub /= sig 
    
    # draw random samples from a bunch of Gaussians 
    X = np.random.normal(0, 1, 3000) 
    X = np.concatenate([X, np.random.normal(-5., 2, 2000)]) 
    X = np.concatenate([X, np.random.normal(4., 0.5, 3000)]) 
    #X = np.concatenate([X, np.random.normal(-1., 10, 3000)]) 
    #X = np.random.normal(0, 1, 100) 
    #X = np.concatenate([X, np.random.normal(-2.5, 2, 200)]) 
    #X = np.concatenate([X, np.random.normal(4., 0.5, 300)]) 
    #X = np.concatenate([X, np.random.normal(-1., 10, 300)]) 

    fig = plt.figure()

    hh = np.histogram(X, normed=True, bins=100, range=[-20, 20])

    Ncomps = [1, 3, 10]
    lstyles = ['-', '--', ':'] 

    xx = np.linspace(-15., 15., 200) 
    X_reshape = X.reshape(1,-1).T
    for i, ncomp in enumerate(Ncomps): 
        sub = fig.add_subplot(3, 1, i+1)
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none') 
        gmm = GMix(n_components=ncomp)
        gmm.fit(X_reshape) 
        
        for icomp in range(len(gmm.means_)):
            gg = gmm.weights_[icomp] * mGauss.pdf(xx, 
                    gmm.means_[icomp][0], gmm.covariances_[icomp][0][0])
            sub.plot(xx, gg, c='k', ls=lstyles[i]) 
        bic = gmm.bic(X_reshape)
        sub.text(0.025, 0.85, r'$N_\mathrm{comp} = '+str(ncomp)+'$', ha='left', va='top', 
                transform=sub.transAxes, fontsize=15)
        sub.text(0.975, 0.85, r'$BIC = '+str(round(bic,2))+'$', ha='right', va='top', 
                transform=sub.transAxes, fontsize=15)

        sub.set_xlim([-15., 15.]) 
        sub.set_ylim([0., 0.35]) 
        if i < 2:
            sub.set_xticklabels([])
        sub.set_yticks([0., 0.2]) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/GMM_pedagog.pdf']), 
            bbox_inches='tight') 
    return None 


def Corner_updatedLike(tag_mcmc, tag_like, ichain): 
    ''' Corner plot with corrected likelihoods. Comparison between the 
    parameter constraints from the pseudo Gaussian likelihood versus the 
    updated likelihood evaluated using importance weights.
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains(tag_mcmc, ichain=ichain)
    # import importance weight
    f_wimp = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), 
        '.', tag_like, '_weights.dat']) 
    wimp = np.loadtxt(f_wimp, skiprows=1, unpack=True, usecols=[2]) 
    lims = np.where(wimp < 1e3) # remove weights particles with non-sensical ratios  

    labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC'] 
    lbltex = [r'$\alpha_\perp$', r'$\alpha_\parallel$', r'$f \sigma_8$', 
            r'$b_1\sigma_8^{NGC}$', r'$b_1\sigma_8^{SGC}$', 
            r'$b_2\sigma_8^{NGC}$', r'$b_2\sigma_8^{SGC}$'] 
    prior_min = [0.8, 0.8, 0.1, 0.3, 0.3, -6., -6.]#, -10000., -10000., 0.5, 0.5]
    prior_max = [1.2, 1.2, 1.1, 2., 2., 6., 6.]#, 10000., 10000., 15., 15.]
    #prior_max = [1.4, 1.4, 1.1, 5., 5., 6., 6.]#, 10000., 10000., 15., 15.]

    nbin = 40 
    chain_arr = np.zeros((len(chain[labels[0]]), len(labels)))
    for i_l, lbl in enumerate(labels):
        chain_arr[:,i_l] = chain[lbl]
    nonweights = np.repeat(np.sum(wimp[lims])/float(chain_arr.shape[0]), chain_arr.shape[0])

    fig = DFM.corner(chain_arr, weights=nonweights, labels=lbltex, quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], range=[(mi,ma) for mi, ma in zip(prior_min, prior_max)],
            smooth=True, bins=nbin, 
            plot_datapoints=False, fill_contours=True, plot_density=False, color='#ee6a50') 
    DFM.corner(chain_arr[lims], weights=wimp[lims], quantiles=[0.16, 0.5, 0.84],
            levels=[0.68, 0.95], range=[(mi,ma) for mi, ma in zip(prior_min, prior_max)],
            smooth=True, bins=nbin, 
            plot_datapoints=False, fill_contours=True, plot_density=False, color='#1F77B4', 
            fig=fig) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/corner.updatedLike.', tag_mcmc, '.', tag_like, '.chain', str(ichain), '.pdf']), 
            bbox_inches='tight') 
    return None


def Like_RSD(): 
    ''' Compare importance sampled likelihoods of GMFs to original likelihood 
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains('beutler_z1', ichain=0)
    # remove burnin?  
    burnin = np.ones(chain['chi2'].shape, dtype=bool) 
    burnin[:int(chain['chi2'].shape[0]/4)] = False 

    # importance weight derived from p_KDE(X_i ICA)
    f_pXiICA = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z1_chain0.RSD_pXiICA_gauss_weights.defICA.kde.dat']) 
    #    'Beutler_et_al_full_shape_analysis_z1_chain0.RSD_pXiICA_gauss_weights.parICA.kde.dat']) 
    wimp = np.loadtxt(f_pXiICA, skiprows=1, unpack=True, usecols=[2]) 
    like_lbl = r'$\prod p_\mathrm{KDE}\left(\textbf{X}_i^\mathrm{ICA}\right)$'
    
    # ignoring some of the nuissance parameters
    labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC'] 
    lbltex = [r'$\alpha_\perp$', r'$\alpha_\parallel$', r'$f \sigma_8$', 
            r'$b_1\sigma_8^{NGC}$', r'$b_1\sigma_8^{SGC}$', 
            r'$b_2\sigma_8^{NGC}$', r'$b_2\sigma_8^{SGC}$'] 
    prior_min = [0.8, 0.8, 0.2, 1., 1., -3., -3.]#, -10000., -10000., 0.5, 0.5]
    prior_max = [1.2, 1.2, 0.8, 1.8, 1.8, 5., 5.]#, 10000., 10000., 15., 15.]
    yrange = [[0.0, 18.], [0., 18.], [0., 10.], [0., 10.], [0., 10.], [0., 0.7], [0., 0.7]]
    yticks = [[0.,2.,4.], [0., 0.5, 1., 1.5, 2.], [0., 0.1, 0.2, 0.3], [0., 2., 4., 6., 8.], [0., 4., 8., 12]]
    nbin = 20 
    
    f_out = open(''.join([UT.tex_dir(), 'dat/pk_likelihood.dat']), 'w') 
    fig = plt.figure(figsize=(5*len(labels), 4)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) 
        # original Beutler et al.(2017) constraints
        hh = np.histogram(chain[labels[i]][burnin], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        lbls = [None, None, None, None, None, 'Beutler et al.(2017)', None]
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=lbls[i]) 
        #sub.plot(bp[0], bp[1], c='k', lw=1, ls=':', label=lbls[i])  
        low, med, high = np.percentile(chain[labels[i]][burnin], [15.86555, 50, 84.13445])
        f_out.write(''.join(['# ', labels[i], ' Beutler et al. (2017)', '\n']))
        f_out.write('\t'.join([str(round(low,5)), str(round(med,5)), str(round(high,5)), '\n']))

        wlim = np.percentile(wimp[burnin], 99.9)
        lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
        # importance weighted constraints
        hh = np.histogram(chain[labels[i]][lims], weights=wimp[lims],normed=True, bins=nbin, 
                range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        lbls = [None, None, None, None, None, None, like_lbl]

        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=lbls[i]) 
        low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

        f_out.write(''.join(['# ', labels[i], ' ', like_lbl, '\n']))
        f_out.write('\t'.join([str(round(low_w,5)), str(round(med_w,5)), str(round(high_w,5)), '\n']))

        f_out.write('\n') 
        if i > 1: sub.legend(loc='upper left', prop={'size': 20})  # legend
        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], labelpad=10, fontsize=25)
        # y-axis
        sub.set_ylim(yrange[i]) 
        #sub.set_yticks(yticks[i]) 

    f_out.close() 
    fig.subplots_adjust(wspace=.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/', 'Like_Pk_comparison.pdf']), bbox_inches='tight') 
    return None


def Like_GMF(): 
    ''' Compare importance sampled likelihoods of GMFs to original likelihood 
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains('manodeep')
    # remove burnin?  
    burnin = np.ones(chain['chi2'].shape, dtype=bool) 
    burnin[:int(chain['chi2'].shape[0]/4)] = False 

    # importance weights derived for N(C_all) estimates 
    f_all = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0', 
        '.gmf_gauss_chi2_weights.dat']) 
    w_all = np.loadtxt(f_all, skiprows=1, unpack=True, usecols=[2]) 
    # importance weight derived from p(X) GMM estimate 
    f_pX = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0.gmf_pX_chi2_weights.gmm_max30comp_bic.dat']) 
    w_pX = np.loadtxt(f_pX, skiprows=1, unpack=True, usecols=[2]) 

    wimps = [w_all, w_pX]
    like_lbls = [r"$\mathcal{N}\left(\,\overline{\zeta},\,{\bf C}^{'} \right)$", 
            r'$p_\mathrm{GMM}\left(\{\zeta^\mathrm{mock}\}\right)$']

    labels = ['logMmin', 'sig_logM', 'logM0', 'logM1', 'alpha']
    lbltex = [r'$\log M_\mathrm{min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\log M_1$', r'$\alpha$'] 
    prior_min = [11.2, 0.001, 6., 12.2, 0.6]
    prior_max = [12.2, 1., 14., 13., 1.2]
    yrange = [[0., 4.], [0., 2.], [0., 0.3], [0., 7], [0., 10]]
    yticks = [[0.,2.,4.], [0., 0.5, 1., 1.5, 2.], [0., 0.1, 0.2, 0.3], [0., 2., 4., 6., 8.], [0., 4., 8., 12]]
    nbin = 20 
    
    f_out = open(''.join([UT.tex_dir(), 'dat/gmf_likelihood.dat']), 'w') 
    fig = plt.figure(figsize=(5*len(labels), 4)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) 
        # original Beutler et al.(2017) constraints
        hh = np.histogram(chain[labels[i]][burnin], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        lbls = [None, None, 'Sinha et al.(2017)', None, None]
        sub.plot(bp[0], bp[1], c='k', lw=1, ls=':', label=lbls[i])  
        low, med, high = np.percentile(chain[labels[i]][burnin], [15.86555, 50, 84.13445])
        f_out.write(''.join(['# ', labels[i], ' Sinha et al. (2017)', '\n']))
        f_out.write('\t'.join([str(round(low,5)), str(round(med,5)), str(round(high,5)), '\n']))

        for iw, wimp, like_lbl in zip(range(len(wimps)), wimps, like_lbls): 
            wlim = np.percentile(wimp[burnin], 99.9)
            lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
            # importance weighted constraints
            hh = np.histogram(chain[labels[i]][lims], weights=wimp[lims],normed=True, bins=nbin, 
                    range=[prior_min[i], prior_max[i]])
            bp = UT.bar_plot(*hh) 
            if iw == 0: lbls = [None, None, None, like_lbl, None]
            else: lbls = [None, None, None, None, like_lbl]

            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=lbls[i]) 
            low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

            f_out.write(''.join(['# ', labels[i], ' ', like_lbl, '\n']))
            f_out.write('\t'.join([str(round(low_w,5)), str(round(med_w,5)), str(round(high_w,5)), '\n']))

        f_out.write('\n') 
        if i > 1: sub.legend(loc='upper right', prop={'size': 20})  # legend
        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], labelpad=10, fontsize=25)
        # y-axis
        sub.set_ylim(yrange[i]) 
        sub.set_yticks(yticks[i]) 

    f_out.close() 
    fig.subplots_adjust(wspace=.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/', 'Like_GMF_comparison.pdf']), bbox_inches='tight') 
    return None
   

def GMF_contours(tag_mcmc='manodeep'):
    ''' Compare 
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains(tag_mcmc)
    # remove burnin?  
    burnin = np.ones(chain['chi2'].shape, dtype=bool) 
    burnin[:int(chain['chi2'].shape[0]/4)] = False 

    # importance weight
    f_all = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0', 
        '.gmf_gauss_chi2_weights.dat']) 
    w_all = np.loadtxt(f_all, skiprows=1, unpack=True, usecols=[2]) 
    f_pX = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0.gmf_pX_chi2_weights.gmm_max30comp_bic.dat']) 
    w_pX = np.loadtxt(f_pX, skiprows=1, unpack=True, usecols=[2]) 

    wimps = [w_all, w_pX]
    imp_lbls = [r"$\mathcal{N} \left(\,\overline{\zeta}, \,{\bf C}^{'} \right)$", #\mathrm{all}\;\theta})$', 
            r'$p_\mathrm{GMM} \left(\{\zeta^\mathrm{mock}\} \right)$']
    imp_colors = ['#1F77B4', '#FF7F0E']
    imp_conts = [False, True]
    imp_lws = [1.25, 0]

    labels = ['logMmin', 'sig_logM', 'logM0', 'logM1', 'alpha']
    lbltex = [r'$\log M_\mathrm{min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\log M_1$', r'$\alpha$'] 
    prior_min = [11., 0.001, 6., 12.3, 0.5]
    prior_max = [12.2, 1., 14., 13.1, 1.5]
    
    # log M_min vs sigma log M and log M_1 vs alpha
    nbin = 20 
    fig = plt.figure(figsize=(2*len(labels), 5)) 
    # log M_min vs sigma log M 
    sub = fig.add_subplot(121)
    DFM.hist2d(chain['logMmin'][burnin], chain['sig_logM'][burnin], color='k', 
            levels=[0.68, 0.95], bins=nbin, 
            range=[[prior_min[0], prior_max[0]], [prior_min[1], prior_max[1]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1, 
            contour_kwargs={'linewidths': 0.5, 'linestyles': 'dotted'}, ax=sub)
    for wimp, imp_color, imp_cont, imp_lw in zip(wimps, imp_colors, imp_conts, imp_lws): 
        wlim = np.percentile(wimp[burnin], 99.9)
        lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
        DFM.hist2d(chain['logMmin'][lims], chain['sig_logM'][lims], weights=wimp[lims], 
                color=imp_color, levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
                range=[[prior_min[0], prior_max[0]], [prior_min[1], prior_max[1]]], 
                plot_datapoints=False, plot_density=False, fill_contours=imp_cont, smooth=1,
                contour_kwargs={'linewidths': imp_lw}, ax=sub)
    sub.set_xticks([11., 11.5, 12.])
    sub.set_xlabel('log $M_\mathrm{min}$', labelpad=10, fontsize=20) 
    sub.set_ylabel('$\sigma_{\mathrm{log} M}$', fontsize=25) 
    # log M_1 vs alpha 
    sub = fig.add_subplot(122)
    DFM.hist2d(chain['logM1'][burnin], chain['alpha'][burnin], color='k', 
            levels=[0.68, 0.95], bins=nbin, 
            range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1, 
            contour_kwargs={'linewidths': 0.5, 'linestyles': 'dotted'}, ax=sub)

    for wimp, imp_color, imp_cont, imp_lw in zip(wimps, imp_colors, imp_conts, imp_lws): 
        wlim = np.percentile(wimp[burnin], 99.9)
        lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
        DFM.hist2d(chain['logM1'][lims], chain['alpha'][lims], weights=wimp[lims], 
                color=imp_color, levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
                range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
                plot_datapoints=False, plot_density=False, fill_contours=imp_cont, smooth=1,
                contour_kwargs={'linewidths': imp_lw}, ax=sub)
    sub.set_xlabel('log $M_1$', labelpad=10, fontsize=20) 
    sub.set_yticks([0.5, 1., 1.5])
    sub.set_ylabel(r'$\alpha$', fontsize=25) 
    
    legs = []
    legs.append(mlines.Line2D([], [], ls='--', c='k', linewidth=2, label='Sinha et al.(2017)'))
    for imp_lbl, imp_color in zip(imp_lbls, imp_colors): 
        legs.append(mlines.Line2D([], [], ls='-', c=imp_color, linewidth=5, alpha=0.5, label=imp_lbl))
    sub.legend(loc='upper right', handles=legs, frameon=False, fontsize=15)#, handletextpad=0.1)#, scatteryoffsets=[0.5])
    fig.subplots_adjust(wspace=.275)
    fig.savefig(''.join([UT.tex_dir(), 'figs/', 
        'GMFcontours_', tag_mcmc, '.pdf']), bbox_inches='tight') 
    return None


if __name__=="__main__": 
    GMM_pedagog()
    #_div_Gauss_gmf(K=10)
    #div_Gauss(K=10)
    #div_nonGauss(K=10)
    #Like_RSD()
    #GMF_contours()
    #Corner_updatedLike('beutler_z1', 'RSD_ica_gauss', 0)
    #Like_RSD('RSD_ica_gauss', ichain=0)
    #Like_RSD('RSD_pca_gauss', ichain=0)
    #Like_GMF('manodeep', 'gmf_lowN_chi2')
    #Like_GMF('manodeep', 'gmf_all_chi2')
    #Like_GMF('manodeep', 'gmf_ica_chi2')
