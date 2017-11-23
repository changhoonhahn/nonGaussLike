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
    - D( mock X || p(X) KDE)
    - D( mock X || p(X) GMM) 
    - D( mock X || PI p(X^i_ICA) KDE)
    - D( mock X || PI p(X^i_ICA) GMM)
    '''
    pretty_colors = prettycolors() 
    fig = plt.figure(figsize=(12,5))
    for i_obv, obv in enumerate(['pk.ngc', 'gmf']): 
        if obv == 'pk.ngc': 
            Nref, ncomp, hrange = 2000, 30, [-0.5, 0.5] #[ 5., 7.]
            #r'$D( \textbf{X}^\mathrm{mock} \parallel p_\mathrm{KDE}(\textbf{X}^\mathrm{mock}))$', 
            #r'$D( \textbf{X}^\mathrm{mock} \parallel \sim p_\mathrm{GMM}(\textbf{X}^\mathrm{mock}))$', 
            #r'$D( \textbf{X}^\mathrm{mock} \parallel \sim \mathcal{N}({\bf C}))$',
            lbls = [None, None,
                    r'$D( \textbf{X}^\mathrm{mock} \parallel \sim \prod p_\mathrm{KDE}(\textbf{X}_i^\mathrm{ICA}))$',
                    r'$D( \textbf{X}^\mathrm{mock} \parallel \sim \prod p_\mathrm{GMM}(\textbf{X}_i^\mathrm{ICA}))$']
            #'pX_KDE.K'+str(K), 'pX_GMM.K'+str(K)+'.ncomp'+str(ncomp), 
            fs = ['ref.K'+str(K), 'pX_gauss.K'+str(K), 'pXi_ICA_scottKDE.K'+str(K), 'pXi_ICA_GMM.K'+str(K)+'.ncomp'+str(ncomp)] 
            hatches = [None, None, '//', '//']

        elif obv == 'gmf': 
            Nref, ncomp, hrange = 10000, 20, [-0.1, 0.25]
            # r'$D( \textbf{X}^\mathrm{mock} \parallel \sim \mathcal{N}({\bf C}))$',
            lbls = [None, None,
                    r'$D( \textbf{X}^\mathrm{mock} \parallel p_\mathrm{KDE}(\textbf{X}^\mathrm{mock}))$', 
                    r'$D( \textbf{X}^\mathrm{mock} \parallel \sim p_\mathrm{GMM}(\textbf{X}^\mathrm{mock}))$'] 
            # 'pXi_ICA_KDE.K'+str(K), 'pXi_ICA_GMM.K'+str(K)+'.ncomp'+str(ncomp)] 
            fs = ['ref.K'+str(K), 'pX_gauss.K'+str(K), 'pX_KDE.K'+str(K), 'pX_GMM.K'+str(K)+'.ncomp'+str(ncomp)]
            hatches = [None, None, '//', '//']

        for i_div, div_func in enumerate(['kl', 'kl']): 
            divs = []  
            for f in fs: 
                f_div = ''.join([UT.dat_dir(), 'diverg.', obv, '.', f, '.Nref', str(Nref), '.', div_func, '.dat']) 
                try: 
                    div = np.loadtxt(f_div)
                except IOError: 
                    continue 
                divs.append(div) 
         
            nbins = 50
            y_max = 0. 
            sub = fig.add_subplot(2,2,2*i_obv+i_div+1)
            for ii, div, lbl in zip(range(len(divs)), divs, lbls): 
                hh = np.histogram(div, normed=True, range=hrange, bins=nbins)
                bp = UT.bar_plot(*hh) 
                if lbl is None: 
                    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
                else: 
                    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], 
                            facecolor='none', edgecolor=pretty_colors[2*ii], linewidth=2,
                            alpha=0.75, hatch=hatches[ii], label=lbl) 
                            #edgecolor='none'
                y_max = max(y_max, bp[1].max()) 
            sub.set_xlim(hrange) 
            sub.set_ylim([0., y_max*1.4]) 

            if i_obv == 0: 
                if i_div == 1: sub.legend(loc='upper right', ncol=2, prop={'size': 15})
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
    fig.subplots_adjust(wspace=.15, hspace=.2)
    f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNNdiverg_nonGauss.pdf'])
    fig.savefig(f_fig, bbox_inches='tight') 
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


def Like_RSD(tag_like, tag_mcmc='beutler_z1', ichain=0):
    ''' comparison between Florian's MCMC posterior distribution from 
    Beutler et al. (2017), to importance sampled posteriors derived for 
    `tag_like`. 
    '''
    if tag_like == 'RSD_ica_gauss': 
        str_like = 'ICA'
    elif tag_like == 'RSD_pca_gauss': 
        str_like = 'PCA'
    else: 
        raise NotImplementedError
    # read in Florian's MCMC chains
    chain = Inf.mcmc_chains(tag_mcmc, ichain=ichain) 
    # read in importance weight
    f_wimp = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), 
        '.', tag_like, '_weights.dat']) 
    wimp = np.loadtxt(f_wimp, skiprows=1, unpack=True, usecols=[2]) 
    
    # remove burnin (half of Florian's chain is burn in) 
    burnin = np.zeros(wimp.shape, dtype=bool) 
    burnin[int(wimp.shape[0]/2):] = True 

    wlim = np.percentile(wimp[burnin], 99.5)
    lims = np.where(burnin & (wimp < wlim)) 
    
    # ignoring some of the nuissance parameters
    labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC'] 
    lbltex = [r'$\alpha_\perp$', r'$\alpha_\parallel$', r'$f \sigma_8$', 
            r'$b_1\sigma_8^{NGC}$', r'$b_1\sigma_8^{SGC}$', 
            r'$b_2\sigma_8^{NGC}$', r'$b_2\sigma_8^{SGC}$'] 
    prior_min = [0.8, 0.8, 0.2, 1., 1., -3., -3.]#, -10000., -10000., 0.5, 0.5]
    prior_max = [1.2, 1.2, 0.8, 1.8, 1.8, 5., 5.]#, 10000., 10000., 15., 15.]
    #prior_max = [1.4, 1.4, 1.1, 5., 5., 6., 6.]#, 10000., 10000., 15., 15.]
    
    nbin = 40 
    fig = plt.figure(figsize=(5*len(labels), 4.5)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) # over-plot the two histograms
        # original Beutler et al.(2017) constraints
        hh = np.histogram(chain[labels[i]][burnin], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', label='Beutler et al.(2017)') 
        # updated constraints
        hh = np.histogram(chain[labels[i]][lims], weights=wimp[lims],normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=str_like+' (imp. sampl.)') 

        # get parameter quanties from the chains and put them in the title of each subplot
        low, med, high = np.percentile(chain[labels[i]], [15.86555, 50, 84.13445])
        low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

        txt = ''.join(['B2017: $', str(round(med,3)), '^{+', str(round(high-med,3)), '}_{-', str(round(med-low,3)), '}$; ', 
            str_like, ': $', str(round(med_w,3)), '^{+', str(round(high_w-med_w,3)), '}_{-', str(round(med_w-low_w,3)), '}$']) 
        sub.set_title(txt)
        #sub.text(0.1, 0.95, txt, ha='left', va='top', transform=sub.transAxes, fontsize=15)
    
        if i == 0: sub.legend(loc='upper right', prop={'size': 15})  # legend
        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], fontsize=25)
        # y-axis
        sub.set_ylim([0., 1.5*hh[0].max()])

    fig.savefig(''.join([UT.tex_dir(), 'figs/Like_RSD.beutler_z1.', tag_like, '.chain', str(ichain), '.pdf']), 
            bbox_inches='tight') 
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
    like_lbls = [r'$\mathcal{N}({\bf C}^{all})', r'$p_\mathrm{GMM}(\{\zeta^{m}\})$']

    labels = ['logMmin', 'sig_logM', 'logM0', 'logM1', 'alpha']
    lbltex = [r'$\log M_\mathrm{min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\log M_1$', r'$\alpha$'] 
    prior_min = [11.2, 0.001, 6., 12.2, 0.6]
    prior_max = [12.2, 1., 14., 13., 1.2]
    yrange = [[0.,4.], [0., 2.], [0., 0.3], [0., 7], [0., 10]]
    nbin = 40 
    
    fig = plt.figure(figsize=(5*len(labels), 4.5)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) 
        # original Beutler et al.(2017) constraints
        hh = np.histogram(chain[labels[i]][burnin], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], facecolor='k', alpha=0.5, edgecolor='none', 
                label='Sinha et al.(2017)') 

        for wimp, like_lbl in zip(wimps, like_lbls): 
            wlim = np.percentile(wimp[burnin], 99.9)
            lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
            # importance weighted constraints
            hh = np.histogram(chain[labels[i]][lims], weights=wimp[lims],normed=True, bins=nbin, 
                    range=[prior_min[i], prior_max[i]])
            bp = UT.bar_plot(*hh) 
            sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=like_lbl+' (imp. sampl.)') 
            # constraints 
            low, med, high = np.percentile(chain[labels[i]], [15.86555, 50, 84.13445])
            low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

            txt = ''.join(['S2017: $', str(round(med,3)), '^{+', str(round(high-med,3)), '}_{-', str(round(med-low,3)), '}$; ', 
                like_lbl, ': $', str(round(med_w,3)), '^{+', str(round(high_w-med_w,3)), '}_{-', str(round(med_w-low_w,3)), '}$']) 
            sub.set_title(txt)

        if i == 0: sub.legend(loc='upper right', prop={'size': 15})  # legend
        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], fontsize=25)
        # y-axis
        sub.set_ylim(yrange[i]) 
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
    imp_lbls = [r'$\mathcal{N}({\bf C}^{\mathrm{all}\;\theta})$', r'$p_\mathrm{GMM}(\{\zeta^{m}\})$']
    imp_colors = ['#1F77B4', '#FF7F0E']
    imp_conts = [False, True]
    imp_lws = [1, 0]

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
            contour_kwargs={'linewidths': 1, 'linestyles': 'dashed'}, ax=sub)
    for wimp, imp_color, imp_cont, imp_lw in zip(wimps, imp_colors, imp_conts, imp_lws): 
        wlim = np.percentile(wimp[burnin], 99.9)
        lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
        DFM.hist2d(chain['logMmin'][lims], chain['sig_logM'][lims], weights=wimp[lims], 
                color=imp_color, levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
                range=[[prior_min[0], prior_max[0]], [prior_min[1], prior_max[1]]], 
                plot_datapoints=False, plot_density=False, fill_contours=imp_cont, smooth=1,
                contour_kwargs={'linewidths': imp_lw}, ax=sub)
    sub.set_xlabel('log $M_\mathrm{min}$', fontsize=20) 
    sub.set_ylabel('$\sigma_{\mathrm{log} M}$', fontsize=20) 
    # log M_1 vs alpha 
    sub = fig.add_subplot(122)
    DFM.hist2d(chain['logM1'][burnin], chain['alpha'][burnin], color='k', 
            levels=[0.68, 0.95], bins=nbin, 
            range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1, 
            contour_kwargs={'linewidths': 1, 'linestyles': 'dashed'}, ax=sub)

    for wimp, imp_color, imp_cont, imp_lw in zip(wimps, imp_colors, imp_conts, imp_lws): 
        wlim = np.percentile(wimp[burnin], 99.9)
        lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)
        DFM.hist2d(chain['logM1'][lims], chain['alpha'][lims], weights=wimp[lims], 
                color=imp_color, levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
                range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
                plot_datapoints=False, plot_density=False, fill_contours=imp_cont, smooth=1,
                contour_kwargs={'linewidths': imp_lw}, ax=sub)
    sub.set_xlabel('log $M_1$', fontsize=20) 
    sub.set_ylabel(r'$\alpha$', fontsize=20) 
    
    legs = []
    legs.append(mlines.Line2D([], [], ls='--', c='k', linewidth=2, label='Sinha+(2017)'))
    for imp_lbl, imp_color in zip(imp_lbls, imp_colors): 
        legs.append(mlines.Line2D([], [], ls='-', c=imp_color, linewidth=5, alpha=0.5, label=imp_lbl))
    sub.legend(loc='upper right', handles=legs, frameon=False, fontsize=15)#, handletextpad=0.1)#, scatteryoffsets=[0.5])
    fig.savefig(''.join([UT.tex_dir(), 'figs/', 
        'GMFcontours.', tag_mcmc, '.pdf']), bbox_inches='tight') 
    return None
   

if __name__=="__main__": 
    #div_Gauss(K=10)
    div_nonGauss(K=10)
    #Like_GMF()
    #GMF_contours()
    #Corner_updatedLike('beutler_z1', 'RSD_ica_gauss', 0)
    #Like_RSD('RSD_ica_gauss', ichain=0)
    #Like_RSD('RSD_pca_gauss', ichain=0)
    #Like_GMF('manodeep', 'gmf_lowN_chi2')
    #Like_GMF('manodeep', 'gmf_all_chi2')
    #Like_GMF('manodeep', 'gmf_ica_chi2')
