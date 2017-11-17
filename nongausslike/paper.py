'''

plots for the paper  


'''
import numpy as np 
import corner as DFM
import wquantiles as wq
from scipy.stats import gaussian_kde as gkde
from sklearn.mixture import GaussianMixture as GMix
from scipy.stats import multivariate_normal as mGauss

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


def divGMF(div_func='renyi:.5', Nref=1000, K=5, n_mc=10, density_method='gkde', n_comp_max=10):
    ''' compare the divergence estimates between 
    D( gmfs || gauss(C_gmf) ), D( gauss(C_gmf) || gauss(C_gmf) ) 
    and 
    D( gmfs || p_ICA ), D( p_ICA || p_ICA )
    '''
    mvn = np.random.multivariate_normal

    # read in mock GMFs from all HOD realizations (20,000 mocks)
    gmfs_mock = NG.X_gmf_all()
    n_mock = gmfs_mock.shape[0] # number of mocks 
    print("%i mocks" % n_mock) 

    gmf_mock_avg = (np.sum(gmfs_mock, axis=0)/float(n_mock)) # average gmf 
    gmfs_white, W = NG.whiten(gmfs_mock - gmf_mock_avg)

    C_gmf = np.cov(gmfs_white.T) # covariance matrix

    # construct a distirbution based on the ICA transform of X_white
    X_ica, _ = NG.Ica(gmfs_white) # get ICA transformation 
    kerns = [] 
    for i_bin in range(X_ica.shape[1]): 
        if density_method == 'gkde': 
            kerns.append(gkde(X_ica[:,i_bin])) 
        elif density_method == 'gmm': 
            gmms, bics = [], [] 
            for i_comp in range(1,n_comp_max+1):
                X = np.reshape(X_ica[:,i_bin], (-1,1))
                gmm = GMix(n_components=i_comp)
                gmm.fit(X) 
                gmms.append(gmm)
                bics.append(gmm.bic(X))
            ibest = np.array(bics).argmin() 
            kerns.append(gmms[ibest])

    # caluclate the divergences now 
    div_knns, div_knns_ref = [], [] 
    div_knns_ica, div_knns_ref_ica = [], [] 
    for i in range(n_mc): 
        print('%i montecarlo' % i)
        # estimate divergence between gmfs_white and a 
        # Gaussian distribution described by C_gmf
        div_knn_i = NG.kNNdiv_gauss(gmfs_white, C_gmf, Knn=K, div_func=div_func, Nref=Nref)

        if i == 0: div_knn = 0. 
        div_knn += div_knn_i 
        div_knns.append(div_knn_i)
        
        # reference divergence in order to showcase the estimator's scatter
        # Gaussian distribution described by C_gmf with same n_mock mocks 
        gauss = mvn(np.zeros(len(gmf_mock_avg)), C_gmf, size=n_mock)
        div_knn_ref_i = NG.kNNdiv_gauss(gauss, C_gmf, Knn=K, div_func=div_func, Nref=Nref)
        div_knns_ref.append(div_knn_ref_i)

        # estimate divergence between the ICA transformed gmfs_white and 
        # distribution derived from KDE of ICA transform
        div_knn_ica_i = NG.kNNdiv_ICA(X_ica, X_ica, Knn=K, div_func=div_func, Nref=Nref, 
                density_method=density_method, n_comp_max=n_comp_max)
        
        if i == 0: div_knn_ica = 0.
        div_knn_ica += div_knn_ica_i
        div_knns_ica.append(div_knn_ica_i)

        # reference divergence in order to showcase the estimator's scatter
        # Gaussian distribution described by C_gmf with same n_mock mocks 
        ica_ref = np.zeros((n_mock, X_ica.shape[1])) 
        for i_bin in range(X_ica.shape[1]): 
            if density_method == 'gkde': 
                ica_ref[:,i_bin] = kerns[i_bin].resample(n_mock)
            elif density_method == 'gmm': 
                samp, _ = kerns[i_bin].sample(n_mock)
                ica_ref[:,i_bin] = samp.flatten()
        div_knn_ref_ica_i = NG.kNNdiv_ICA(ica_ref, X_ica, Knn=K, div_func=div_func, Nref=Nref, 
                density_method=density_method, n_comp_max=n_comp_max) 
        div_knns_ref_ica.append(div_knn_ref_ica_i)

    div_knn /= float(n_mc)
    div_knn_ica /= float(n_mc)

    fig = plt.figure(figsize=(7,5))
    sub = fig.add_subplot(211)
    # divergence between mock GMFs and Gaussian distribution 
    hh = np.histogram(np.array(div_knns), normed=True)
    bp = UT.bar_plot(*hh) 
    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', 
            label=r'$D($mock $\zeta(N)\parallel \mathcal{N})$') 
    # reference to show scatter of estimator
    hh = np.histogram(np.array(div_knns_ref), normed=True)
    bp = UT.bar_plot(*hh) 
    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
    sub.set_xlim([-0.05, 0.25]) 
    sub.legend(loc='upper right', prop={'size': 15})
        
    sub = fig.add_subplot(212)
    # divergence between mock GMFs and Gaussian distribution 
    hh = np.histogram(np.array(div_knns_ica), normed=True)
    bp = UT.bar_plot(*hh) 
    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', 
            label=r'$D($mock $\zeta^\mathrm{ICA}(N)\parallel p^\mathrm{ICA})$') 
    # reference to show scatter of estimator
    hh = np.histogram(np.array(div_knns_ref_ica), normed=True)
    bp = UT.bar_plot(*hh) 
    sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none') 
    sub.set_xlim([-0.05, 0.25]) 
    sub.legend(loc='upper right', prop={'size': 15})
    if 'renyi' in div_func: 
        alpha = float(div_func.split(':')[-1])
        sub.set_xlabel(r'Renyi-$\alpha='+str(alpha)+'$ divergence', fontsize=20)
    elif 'kl' in div_func: 
        sub.set_xlabel(r'KL divergence', fontsize=20)

    if 'renyi' in div_func: 
        f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNN_divergence.gmf.renyi', str(alpha), '.', 
            density_method, '.pdf'])
    elif div_func == 'kl':
        f_fig = ''.join([UT.tex_dir(), 'figs/', 'kNN_divergence.gmf.kl.', 
            density_method, '.pdf'])
    fig.savefig(f_fig)#, bbox_inches='tight') 
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


def Like_GMF(tag_mcmc, tag_like):
    '''
    '''
    if tag_like == 'gmf_ica_chi2': 
        str_like = 'ICA'
    elif tag_like == 'gmf_pca_chi2': 
        str_like = 'PCA'
    elif tag_like == 'gmf_all_chi2': 
        str_like = r'all $\theta$s'
    elif tag_like == 'gmf_lowN_chi2': 
        str_like = r'low $N$'
    elif tag_like == 'gmf_gauss_chi2': 
        str_like = 'Gauss'
    else: 
        raise NotImplementedError
    # import MCMC chain 
    chain = Inf.mcmc_chains(tag_mcmc)

    # import importance weight
    f_wimp = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0', 
        '.', tag_like, '_weights.dat']) 
    wimp = np.loadtxt(f_wimp, skiprows=1, unpack=True, usecols=[2]) 
    
    # remove burnin?  
    burnin = np.ones(wimp.shape, dtype=bool) 
    burnin[:int(wimp.shape[0]/4)] = False 

    wlim = np.percentile(wimp[burnin], 99.9)
    lims = np.where(burnin & (wimp < wlim)) #lims = np.where(wimp < 1e3)

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
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], edgecolor='none', label='Sinha et al.(2017)') 
        # updated constraints
        hh = np.histogram(chain[labels[i]][lims], weights=wimp[lims],normed=True, bins=nbin, range=[prior_min[i], prior_max[i]])
        bp = UT.bar_plot(*hh) 
        sub.fill_between(bp[0], np.zeros(len(bp[0])), bp[1], alpha=0.75, edgecolor='none', label=str_like+' (imp. sampl.)') 
        # constraints 
        low, med, high = np.percentile(chain[labels[i]], [15.86555, 50, 84.13445])
        low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

        txt = ''.join(['S2017: $', str(round(med,3)), '^{+', str(round(high-med,3)), '}_{-', str(round(med-low,3)), '}$; ', 
            str_like, ': $', str(round(med_w,3)), '^{+', str(round(high_w-med_w,3)), '}_{-', str(round(med_w-low_w,3)), '}$']) 
        sub.set_title(txt)

        if i == 0: sub.legend(loc='upper right', prop={'size': 15})  # legend
        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], fontsize=25)
        # y-axis
        sub.set_ylim(yrange[i]) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/', 
        'Like_GMF.', tag_mcmc, '.', tag_like, '.pdf']), bbox_inches='tight') 
    return None
   

def GMF_contours(tag_mcmc='manodeep'):
    ''' Compare 
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains(tag_mcmc)

    # import importance weight
    f_all = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0', 
        '.gmf_gauss_chi2_weights.dat']) 
    w_all = np.loadtxt(f_all, skiprows=1, unpack=True, usecols=[2]) 
    f_ica = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0', 
        '.gmf_ica_chi2_weights.dat']) 
    w_ica = np.loadtxt(f_ica, skiprows=1, unpack=True, usecols=[2]) 
    
    # remove burnin?  
    burnin = np.ones(w_all.shape, dtype=bool) 
    burnin[:int(w_all.shape[0]/4)] = False 

    wlim_all = np.percentile(w_all[burnin], 99.9)
    lims_all = np.where(burnin & (w_all < wlim_all)) #lims = np.where(wimp < 1e3)
    wlim_ica = np.percentile(w_ica[burnin], 99.9)
    lims_ica = np.where(burnin & (w_ica < wlim_ica)) #lims = np.where(wimp < 1e3)

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

    DFM.hist2d(chain['logMmin'][lims_all], chain['sig_logM'][lims_all], weights=w_all[lims_all], 
            color='#1F77B4', levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
            range=[[prior_min[0], prior_max[0]], [prior_min[1], prior_max[1]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1,
            contour_kwargs={'linewidths': 1}, ax=sub)
    
    DFM.hist2d(chain['logMmin'][lims_ica], chain['sig_logM'][lims_ica], weights=w_ica[lims_ica], 
            color='#FF7F0E', levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
            range=[[prior_min[0], prior_max[0]], [prior_min[1], prior_max[1]]], 
            plot_datapoints=False, plot_density=False, fill_contours=True, smooth=1,
            contour_kwargs={'linewidths': 0.0}, ax=sub)
    sub.set_xlabel('log $M_\mathrm{min}$', fontsize=20) 
    sub.set_ylabel('$\sigma_{\mathrm{log} M}$', fontsize=20) 
    # log M_1 vs alpha 
    sub = fig.add_subplot(122)

    DFM.hist2d(chain['logM1'][burnin], chain['alpha'][burnin], color='k', 
            levels=[0.68, 0.95], bins=nbin, 
            range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1, 
            contour_kwargs={'linewidths': 1, 'linestyles': 'dashed'}, ax=sub)

    DFM.hist2d(chain['logM1'][lims_all], chain['alpha'][lims_all], weights=w_all[lims_all], 
            color='#1F77B4', levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
            range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
            plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1,
            contour_kwargs={'linewidths': 1}, ax=sub)
    
    DFM.hist2d(chain['logM1'][lims_ica], chain['alpha'][lims_ica], weights=w_ica[lims_ica], 
            color='#FF7F0E', levels=[0.68, 0.95], alpha=0.1, bins=nbin, 
            range=[[prior_min[3], prior_max[3]], [prior_min[4], prior_max[4]]], 
            plot_datapoints=False, plot_density=False, fill_contours=True, smooth=1,
            contour_kwargs={'linewidths': 0.0}, ax=sub)
    sub.set_xlabel('log $M_1$', fontsize=20) 
    sub.set_ylabel(r'$\alpha$', fontsize=20) 

    leg_sinha = mlines.Line2D([], [], ls='--', c='k', linewidth=2, 
            label='Sinha+(2017)' )
    leg_all = mlines.Line2D([], [], ls='-', c='#1F77B4', linewidth=2, alpha=0.5, 
            label=r'${\bf C}^{\mathrm{all}\;\theta}$')
    leg_ica = mlines.Line2D([], [], ls='-', c='#FF7F0E', linewidth=12, alpha=0.5,
            label='ICA')
    sub.legend(loc='upper right', handles=[leg_sinha, leg_all, leg_ica], 
            frameon=False, fontsize=15)#, handletextpad=0.1)#, scatteryoffsets=[0.5])

    fig.savefig(''.join([UT.tex_dir(), 'figs/', 
        'GMFcontours.', tag_mcmc, '.pdf']), bbox_inches='tight') 
    return None
   

if __name__=="__main__": 
    divGMF(div_func='kl', Nref=10000, K=10, n_mc=5, density_method='gmm')
    divGMF(div_func='kl', Nref=10000, K=10, n_mc=5, density_method='gkde')
    #Corner_updatedLike('beutler_z1', 'RSD_ica_gauss', 0)
    #Like_RSD('RSD_ica_gauss', ichain=0)
    #Like_RSD('RSD_pca_gauss', ichain=0)
    #Like_GMF('manodeep', 'gmf_lowN_chi2')
    #Like_GMF('manodeep', 'gmf_all_chi2')
    #Like_GMF('manodeep', 'gmf_ica_chi2')
    #GMF_contours(tag_mcmc='manodeep')
