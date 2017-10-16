'''
'''
import numpy as np 

import env
import util as UT 
import model as Mod
import infer as Inf

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

    k0, p0k = Inf.data(0, 1, 'ngc')
    k2, p2k = Inf.data(2, 1, 'ngc')
    k4, p4k = Inf.data(4, 1, 'ngc')
    k = np.concatenate([k0, k2, k4])

    # alpha_perp, alpha_para, fsig8, b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, NNGC, NSGC, sigmavNGC, sigmavSGC
    #value_array = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 1.16, -1580., -1580., 6.1, 6.1]
    value_array = [1.008, 1.001, 0.478, 1.339, 1.337, 1.16, 0.32, -1580., -930., 6.1, 6.8]
    #value_array = [1.00830111426, 1.0007368972, 0.478098423689, 1.33908539185, 1.33663505549, 1.15627984704, 0.31657562682, -1580.01689181, -928.488535962, 6.14815801563, 6.79551199595] #z1 max likelihood
    modelX = Mod.taruya_model_module_combined_win_local(100, binrange1, binrange2, binrange3, maxbin1/binsize, 
            k, value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], 
            value_array[6], value_array[7], value_array[8], value_array[9], value_array[10])
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

    k0, p0k = Inf.data(0, 1, 'sgc')
    k2, p2k = Inf.data(2, 1, 'sgc')
    k4, p4k = Inf.data(4, 1, 'sgc')
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


if __name__=="__main__":
    print(lnPrior())
