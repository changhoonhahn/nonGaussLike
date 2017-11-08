'''

plots for the paper  


'''
import numpy as np 
import corner as DFM
import wquantiles as wq
from scipy.stats import gaussian_kde as gkde
from scipy.stats import multivariate_normal as mGauss

# -- local -- 
import data as Data
import util as UT 
import infer as Inf
import nongauss as NG 

# -- plotting -- 
from ChangTools.plotting import prettycolors
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


def Like_RSD(tag_like, tag_mcmc='beutler_z1'):
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
    chain, wimp = [], [] 
    for ichain in range(3) 
        # read in Florian's MCMC chains
        chain.append(Inf.mcmc_chains(tag_mcmc, ichain=ichain)) 
        # read in importance weight
        f_wimp = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
            'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), 
            '.', tag_like, '_weights.dat']) 
        wimp.append(np.loadtxt(f_wimp, skiprows=1, unpack=True, usecols=[2])) 
    chain = np.concatenate(chain) 
    wimp = np.concatenate(wimp) 
    print chain.shape
    print wimp.shape 
    lims = np.where(wimp < 1e3)

    
    # ignoring some of the nuissance parameters
    labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC'] 
    lbltex = [r'$\alpha_\perp$', r'$\alpha_\parallel$', r'$f \sigma_8$', 
            r'$b_1\sigma_8^{NGC}$', r'$b_1\sigma_8^{SGC}$', 
            r'$b_2\sigma_8^{NGC}$', r'$b_2\sigma_8^{SGC}$'] 
    prior_min = [0.8, 0.8, 0.2, 1., 1., -3., -3.]#, -10000., -10000., 0.5, 0.5]
    prior_max = [1.2, 1.2, 0.8, 1.8, 1.8, 5., 5.]#, 10000., 10000., 15., 15.]
    #prior_max = [1.4, 1.4, 1.1, 5., 5., 6., 6.]#, 10000., 10000., 15., 15.]
    
    nbin = 40 
    fig = plt.figure(figsize=(5*len(labels), 5)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) 
        # over-plot the two histograms
        hh = sub.hist(chain[labels[i]], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]],
                alpha=1, edgecolor='none', label='Beutler et al.(2017)') 
        sub.hist(chain[labels[i]][lims], weights=wimp[lims], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]], 
                alpha=0.75, edgecolor='none', label=str_like+' (imp. sampl.)') 

        # get parameter quanties from the chains and put them in the title of each subplot
        low, med, high = np.percentile(chain[labels[i]], [15.86555, 50, 84.13445])
        low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]

        txt = ''.join(['B2017: $', str(round(med,3)), '^{+', str(round(high-med,3)), '}_{-', str(round(med-low,3)), '}$\n', 
            str_like, ': $', str(round(med_w,3)), '^{+', str(round(high_w-med_w,3)), '}_{-', str(round(med_w-low_w,3)), '}$']) 
        sub.set_title(txt)
        #sub.text(0.1, 0.95, txt, ha='left', va='top', transform=sub.transAxes, fontsize=15)
    
        # legend
        if i == 0: sub.legend(loc='upper right', prop={'size': 15})  

        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], fontsize=20)
        # y-axis
        sub.set_ylim([0., 1.5*hh[0].max()])

    fig.savefig(''.join([UT.tex_dir(), 'figs/Like_RSD.beutler_z1.', tag_like, '.pdf']), 
            bbox_inches='tight') 
    return None


def LikelihoodPDF_GMF(tag_mcmc, tag_like, irun):
    '''
    '''
    # import MCMC chain 
    chain = Inf.mcmc_chains(tag_mcmc)

    # import importance weight
    f_wimp = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0'
        '.run', str(irun), '.', tag_like, '_weights.dat']) 
    wimp = np.loadtxt(f_wimp, skiprows=1, unpack=True, usecols=[2]) 
    lims = np.where(wimp < 1e3)

    labels = ['logMmin', 'sig_logM', 'logM0', 'logM1', 'alpha']
    lbltex = [r'$\log M_\mathrm{min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\log M_1$', r'$\alpha$'] 
    prior_min = [11., 0.001, 6., 12., 0.001]
    prior_max = [12.2, 1., 14., 14., 2.]
    
    nbin = 40 
    fig = plt.figure(figsize=(5*len(labels), 5)) 
    for i in range(len(labels)): 
        sub = fig.add_subplot(1, len(labels), i+1) 
        hh = sub.hist(chain[labels[i]], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]],
                alpha=0.5, label='Sinha+2017') 
        sub.hist(chain[labels[i]][lims], weights=wimp[lims], normed=True, bins=nbin, range=[prior_min[i], prior_max[i]], 
                alpha=0.5) 
        # constraints 
        low, med, high = np.percentile(chain[labels[i]], [15.86555, 50, 84.13445])
        low_w, med_w, high_w = [wq.quantile_1D(chain[labels[i]][lims], wimp[lims], qq) for qq in [0.1586555, 0.50, 0.8413445]]
        txt = ''.join(['B2017:$', str(round(med,3)), '^{+', str(round(high-med,3)), '}_{-', str(round(med-low,3)), '}$\n', 
            '$', str(round(med_w,3)), '^{+', str(round(high_w-med_w,3)), '}_{-', str(round(med_w-low_w,3)), '}$']) 
        sub.text(0.1, 0.95, txt, 
                ha='left', va='top', transform=sub.transAxes, fontsize=20)

        # x-axis 
        sub.set_xlim([prior_min[i], prior_max[i]]) 
        sub.set_xlabel(lbltex[i], fontsize=20)
        # y-axis
        sub.set_ylim([0., 1.5*hh[0].max()])
        #if i == 0: sub.legend(loc='upper right', prop={'size':15}) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/likelihoodPDF_gmf.', tag_mcmc, '.', tag_like, '.run', str(irun), '.pdf']), 
            bbox_inches='tight') 
    return None
   

if __name__=="__main__": 
    #Corner_updatedLike('beutler_z1', 'RSD_ica_gauss', 0)
    Like_RSD('RSD_ica_gauss', 0)
    #LikelihoodPDF_GMF('manodeep', 'gmf_ica_chi2', 1)
    #LikelihoodPDF_GMF('manodeep', 'gmf_pca_chi2', 1)
