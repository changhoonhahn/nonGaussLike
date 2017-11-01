'''
'''
import time
import numpy as np 
import corner as DFM

import env
import util as UT 
import data as Dat
import model as Mod
import infer as Inf
import nongauss as NG

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


def importance_weight():  
    '''
    '''
    # read in BOSS P(k) (read in data D) 
    pkay = Dat.Pk() 
    k_list, pk_ngc_data = [], []
    for ell in [0,2,4]: 
        k, plk_ngc = pkay.Observation(ell, 1, 'ngc')
        k_list.append(k)
        pk_ngc_data.append(plk_ngc)
    binrange1, binrange2, binrange3 = len(k_list[0]), len(k_list[1]), len(k_list[2])
    maxbin1 = len(k_list[0])+1
    k = np.concatenate(k_list)
    
    chain = Inf.mcmc_chains('beutler_z1')
    
    # calculate D - m(theta) for all the mcmc chain
    delta_ngc = []
    for i in range(10):#len(chain['chi2'])): 
        model_i = Mod.taruya_model(100, binrange1, binrange2, binrange3, maxbin1, k, 
                chain['alpha_perp'][i], chain['alpha_para'][i], 
                chain['fsig8'][i], 
                chain['b1sig8_NGC'][i], chain['b1sig8_SGC'][i], 
                chain['b2sig8_NGC'][i], chain['b2sig8_SGC'][i], 
                chain['N_NGC'][i], chain['N_SGC'][i], 
                chain['sigmav_NGC'][i], chain['sigmav_SGC'][i])
        delta_ngc.append(model_i[0] - np.concatenate(pk_ngc_data))
        
    # import PATCHY mocks 
    pk_ngc_list = [] 
    for ell in [0, 2, 4]:
        if ell == 4: kmax = 0.1 
        else: kmax = 0.15
        pk_ngc_list.append(NG.X_pk('patchy.ngc.z1', krange=[0.01,kmax], ell=ell, sys='fc'))
    pk_ngc_mock = np.concatenate(pk_ngc_list, axis=1) 

    lnP_ica_ngc = NG.lnL_ica(np.array(delta_ngc), pk_ngc_mock) 
    lnP_pca_ngc = NG.lnL_pca(np.array(delta_ngc), pk_ngc_mock) 
    
    lnw_ngc = lnP_ica_ngc - lnP_pca_ngc
    print  np.exp(lnw_ngc)
    return None


def model():
    kbins = 60
    binsize = 120/kbins
    minbin1 = 2
    minbin2 = 2
    minbin3 = 2
    maxbin1 = 30
    maxbin2 = 30
    maxbin3 = 20
    binrange1 = int(0.5*(maxbin1-minbin1))
    binrange2 = int(0.5*(maxbin2-minbin2))
    binrange3 = int(0.5*(maxbin3-minbin3))
    totbinrange = binrange1+binrange2+binrange3
    
    pkay = Dat.Pk() 
    k0, p0k = pkay.Observation(0, 1, 'ngc')
    k2, p2k = pkay.Observation(2, 1, 'ngc')
    k4, p4k = pkay.Observation(4, 1, 'ngc')
    k = np.concatenate([k0, k2, k4])

    # alpha_perp, alpha_para, fsig8, b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, NNGC, NSGC, sigmavNGC, sigmavSGC
    #value_array = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 1.16, -1580., -1580., 6.1, 6.1]
    value_array = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 0.32, -1580., -930., 6.1, 6.8]
    #value_array = [1.00830111426, 1.0007368972, 0.478098423689, 1.33908539185, 1.33663505549, 1.15627984704, 0.31657562682, -1580.01689181, -928.488535962, 6.14815801563, 6.79551199595] #z1 max likelihood
    t_start = time.time() 
    modelX = Mod.taruya_model(100, binrange1, binrange2, binrange3, maxbin1/binsize, 
            k, value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], 
            value_array[6], value_array[7], value_array[8], value_array[9], value_array[10])
    print time.time() - t_start, ' seconds'
    model_ngc = modelX[0]
    model_sgc = modelX[1]

    # read in pre-window convlution 
    k_nw, p0k_ngc_nw, p2k_ngc_nw, p4k_ngc_nw, p0k_sgc_nw, p2k_sgc_nw, p4k_sgc_nw = np.loadtxt(''.join([UT.dat_dir(), 'boss/test.nowindow.dat']), unpack=True) 

    # now plot 
    pretty_colors = prettycolors()
    fig = plt.figure(figsize=(11,5)) 
    sub = fig.add_subplot(121)
    sub.scatter(k0, k0*p0k, c=pretty_colors[1], lw=0)
    sub.plot(k0, k0*model_ngc[:binrange1], c=pretty_colors[1])
    sub.plot(k_nw, k_nw*p0k_ngc_nw, c=pretty_colors[1], lw=1, ls='--')
    sub.scatter(k2, k2*p2k, c=pretty_colors[3], lw=0)
    sub.plot(k_nw, k_nw*p2k_ngc_nw, c=pretty_colors[3], lw=1, ls='--')
    sub.plot(k2, k2*model_ngc[binrange1:binrange1+binrange2], c=pretty_colors[3])
    sub.scatter(k4, k4*p4k, c=pretty_colors[5], lw=0)
    sub.plot(k_nw, k_nw*p4k_ngc_nw, c=pretty_colors[5], lw=1, ls='--')
    sub.plot(k4, k4*model_ngc[binrange1+binrange2:totbinrange], c=pretty_colors[5])
    sub.text(0.9, 0.05, 'NGC',
            ha='right', va='bottom', transform=sub.transAxes, fontsize=20)

    sub.set_xlim([0.01, 0.15]) 
    sub.set_ylim([-750, 2250])
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_ylabel('$k \, P_{\ell}(k)$', fontsize=25) 
    
    k0, p0k = pkay.Observation(0, 1, 'sgc')
    k2, p2k = pkay.Observation(2, 1, 'sgc')
    k4, p4k = pkay.Observation(4, 1, 'sgc')
    k = np.concatenate([k0, k2, k4])
    
    sub = fig.add_subplot(122)
    sub.scatter(k0, k0*p0k, c=pretty_colors[1], lw=0)
    sub.plot(k0, k0*model_sgc[:binrange1], c=pretty_colors[1])
    sub.plot(k_nw, k_nw*p0k_sgc_nw, c=pretty_colors[1], lw=1, ls='--')
    sub.scatter(k2, k2*p2k, c=pretty_colors[3], lw=0)
    sub.plot(k2, k2*model_sgc[binrange1:binrange1+binrange2], c=pretty_colors[3])
    sub.plot(k_nw, k_nw*p2k_sgc_nw, c=pretty_colors[3], lw=1, ls='--')
    sub.scatter(k4, k4*p4k, c=pretty_colors[5], lw=0)
    sub.plot(k4, k4*model_sgc[binrange1+binrange2:totbinrange], c=pretty_colors[5])
    sub.plot(k_nw, k_nw*p4k_sgc_nw, c=pretty_colors[5], lw=1, ls='--')
    sub.text(0.9, 0.05, 'SGC',
            ha='right', va='bottom', transform=sub.transAxes, fontsize=20)

    sub.set_xlim([0.01, 0.15]) 
    sub.set_ylim([-750, 2250])
    sub.set_xlabel('$k$', fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'tests/test.model.plk.png']), bbox_inches='tight') 
    return None 


def lnPrior(): 
    ''' ***TESTED***
    test the prior function 
    '''
    theta = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 0.32, -1580., -930., 6.1, 6.8]
    return Inf.lnPrior(theta)


def lnPost(zbin=1): 
    ''' *** TESTED *** 
    Likelihood and posterior functions can reproduce the chi-squared from 
    Florian's MCMC chain.

    Test that the ln(Posterior) reproduces the chi-squared values of 
    Florian's MCMC chains 
    ''' 
    # read in Florian's chains
    chain_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z', str(zbin), '_chain', str(0), '.dat']) 
    sample = np.loadtxt(chain_file, skiprows=1)
    chi2s = sample[:,-1]
    sample = sample[:,1:-1]

    # read in BOSS P(k) NGC + SGC
    pkay = Dat.Pk() 
    k0, p0k_ngc = pkay.Observation(0, zbin, 'ngc')
    k2, p2k_ngc = pkay.Observation(2, zbin, 'ngc')
    k4, p4k_ngc = pkay.Observation(4, zbin, 'ngc')
    k0, p0k_sgc = pkay.Observation(0, zbin, 'sgc')
    k2, p2k_sgc = pkay.Observation(2, zbin, 'sgc')
    k4, p4k_sgc = pkay.Observation(4, zbin, 'sgc')
    k_list = [k0, k2, k4]
    pk_ngc_list = [p0k_ngc, p2k_ngc, p4k_ngc]
    pk_sgc_list = [p0k_sgc, p2k_sgc, p4k_sgc]

    # read in Covariance matrix 
    # currently for testing purposes, 
    # implemented to read in Florian's covariance matrix  
    _, _, C_pk_ngc = Dat.beutlerCov(zbin, NorS='ngc', ell='all')
    _, _, C_pk_sgc = Dat.beutlerCov(zbin, NorS='sgc', ell='all')

    # calculate precision matrices (including the hartlap factor) 
    n_mocks_ngc = 2045
    n_mocks_sgc = 2048
    f_hartlap_ngc = (float(n_mocks_ngc) - float(len(np.concatenate(pk_ngc_list))) - 2.)/(float(n_mocks_ngc) - 1.)
    f_hartlap_sgc = (float(n_mocks_sgc) - float(len(np.concatenate(pk_sgc_list))) - 2.)/(float(n_mocks_sgc) - 1.)

    Cinv_ngc = np.linalg.inv(C_pk_ngc) 
    Cinv_sgc = np.linalg.inv(C_pk_sgc)
    Cinv_ngc *= f_hartlap_ngc 
    Cinv_sgc *= f_hartlap_sgc

    lnpost_args = (k_list, pk_ngc_list, pk_sgc_list, Cinv_ngc, Cinv_sgc)
    
    for i in range(10): 
        print 'Like', -2.*Inf.lnLike(sample[i,:], *lnpost_args)
        print 'Post', -2.*Inf.lnPost(sample[i,:], *lnpost_args)
        print 'chi2', chi2s[i]
    return None


def plot_mcmc(tag=None, zbin=1, nwalkers=48, nchains_burn=200):
    '''
    Plot MCMC chains
    '''
    labels = [r'$\alpha_\perp$', r'$\alpha_\parallel$', '$f\sigma_8$', 
            '$b_1^\mathrm{NGC} \sigma_8$', '$b_1^\mathrm{SGC}\sigma_8$', 
            '$b_2^\mathrm{NGC} \sigma_8$', '$b_2^\mathrm{SGC}\sigma_8$', 
            '$N_\mathrm{NGC}$', '$N_\mathrm{SGC}$', '$\sigma_v^\mathrm{NGC}$', '$\sigma_v^\mathrm{SGC}$']
    # plot range are the prior ranges 
    prior_min = [0.8, 0.8, 0.1, 0.3, 0.3, -6., -6., -10000., -10000., 0.5, 0.5]
    prior_max = [1.4, 1.4, 1.1, 5., 5., 6., 6., 10000., 10000., 15., 15.]
    plot_range = np.zeros((len(prior_min),2))
    plot_range[:,0] = prior_min
    plot_range[:,1] = prior_max

    # chain file
    if tag == 'beutler': 
        chain_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
            'Beutler_et_al_full_shape_analysis_z', str(zbin), '_chain', str(0), '.dat']) 
        sample = np.loadtxt(chain_file, skiprows=1)
        sample = sample[:,1:-1]
    else: 
        chain_file = ''.join([UT.dat_dir(), 'mcmc/', tag, '.chain', str(0), 
            '.zbin', str(zbin), '.dat']) 
        sample = np.loadtxt(chain_file)
    print len(labels)
    print sample.shape
    
    # Posterior Likelihood Corner Plot
    fig = DFM.corner(
            sample[nchains_burn*nwalkers:],
            truth_color='#ee6a50',
            labels=labels,
            label_kwargs={'fontsize': 25},
            range=plot_range,
            quantiles=[0.16,0.5,0.84],
            show_titles=True,
            title_args={"fontsize": 12},
            fill_contours=True,
            levels=[0.68, 0.95],
            color='b', bins=16, smooth=1.0)

    fig_file = ''.join([UT.fig_dir(), 'mcmc.corner.', tag, '.chain', str(0), 
        '.zbin', str(zbin), '.png']) 
    plt.savefig(fig_file)
    plt.close()
    raise ValueError
     
    # MCMC Chain plot
    Ndim = len(sample[0])
    Nchain = len(sample)/Nwalkers

    chain_ensemble = sample.reshape(Nchain, Nwalkers, Ndim)
    fig , axes = plt.subplots(5, 1 , sharex=True, figsize=(10, 12))
    for i in xrange(5):
        axes[i].plot(chain_ensemble[:, :, i], color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].axhline(truths[i], color="#888888", lw=2)
        axes[i].vlines(Nchains_burn, plot_range[i,0], plot_range[i,1], colors='#ee6a50', linewidth=4, alpha=1)
        axes[i].set_ylim([plot_range[i,0], plot_range[i,1]])
        axes[i].set_xlim(0, 6000)
        axes[i].set_ylabel(labels[i], fontsize=25)

    axes[4].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = ''.join([UT.fig_dir(), 'mcmc.chain.', tag, '.chain', str(0), 
        '.zbin', str(zbin), '.png']) 
    plt.savefig(fig_file)
    plt.close()


if __name__=="__main__":
    model() 
    #importance_weight()
    #lnPost()
    #plot_mcmc(tag='beutler', zbin=1, nwalkers=48, nchains_burn=200) #tag=testing, zbin=1)
