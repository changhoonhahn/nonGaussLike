'''

Generate importance weights for different re-evaluations of the
likelihood. 

'''
import sys as Sys
import numpy as np 

import env 
import util as UT 
import infer as Inf


def importance_weight_Pk(tag_like, ichain, zbin=1, ica_algorithm=None, 
        density_method='kde', n_comp_max=20, info_crit='bic', njobs=1): 
    ''' save importance weights to file 
    '''
    # read in MCMC chain 
    chain = Inf.mcmc_chains('beutler_z'+str(zbin), ichain=ichain)
    ws = Inf.W_importance(tag_like, chain, zbin=zbin, ica_algorithm=ica_algorithm,
            density_method=density_method, info_crit=info_crit, njobs=njobs)

    if density_method == 'kde': 
        str_density = '.kde'
    elif density_method == 'gmm': 
        str_density = ''.join(['.gmm_max', str(n_comp_max), 'comp_', info_crit]) 
    
    str_ica = ''
    if ica_algorithm is not None: 
        str_ica = ''.join(['.', ica_algorithm[:3], 'ICA']) 

    weight_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z', str(zbin), '_chain', str(ichain), 
        '.', tag_like, '_weights', str_ica, str_density, '.dat']) 
    hdr = 'ln(P_denom), ln(P_numer), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


def importance_weight_Gmf(tag_like, ica_algorithm=None,
        density_method='kde', n_comp_max=20, info_crit='bic', njobs=1): 
    ''' save importance weights for Manodeep's MCMC chain to file 
    '''
    chain = Inf.mcmc_chains('manodeep')
    ws = Inf.W_importance(tag_like, chain, ica_algorithm=ica_algorithm,
            density_method=density_method, info_crit=info_crit, njobs=njobs)
    
    if density_method == 'kde': 
        str_density = '.kde'
    elif density_method == 'gmm': 
        str_density = ''.join(['.gmm_max', str(n_comp_max), 'comp_', info_crit]) 
    
    str_ica = ''
    if ica_algorithm is not None: 
        str_ica = ''.join(['.', ica_algorithm[:3], 'ICA']) 

    weight_file = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0',
        '.', tag_like, '_weights', str_ica, str_density, '.dat']) 

    hdr = 'ln(P_denom), ln(P_numer), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


if __name__=='__main__': 
    # e.g. python run/run_importance_weight.py gmf_pca_chi2 manodeep
    analysis = Sys.argv[1] 
    tag_like = Sys.argv[2]
    print('--- %s, %s ---' % (analysis, tag_like))
    i = 3 
    if 'ica' in tag_like.lower(): 
        tag_ica = Sys.argv[i]
        i += 1 
    else: 
        tag_ica = None
    print(tag_ica)
    density = Sys.argv[i] 
    if density == 'kde': 
        njobs = int(Sys.argv[i+1])
    elif density == 'gmm': 
        ncompmax = int(Sys.argv[i+1])
        infocrit = Sys.argv[i+2]
        njobs = int(Sys.argv[i+3]) 

    if analysis == 'pk': # Florian's analysis 
        if density == 'kde':  
            importance_weight_Pk(tag_like, 0, zbin=1, ica_algorithm=tag_ica, 
                    density_method=density, njobs=njobs)
        elif density == 'gmm': 
            importance_weight_Pk(tag_like, 0, zbin=1, ica_algorithm=tag_ica,
                    density_method=density, n_comp_max=ncompmax, info_crit=infocrit, njobs=njobs) 
    elif analysis == 'gmf': # manodeeps analysis
        if density == 'kde': 
            importance_weight_Gmf(tag_like, ica_algorithm=tag_ica, 
                    density_method=density, njobs=njobs) 
        elif density == 'gmm': 
            importance_weight_Gmf(tag_like, ica_algorithm=tag_ica,
                    density_method=density, n_comp_max=ncompmax, info_crit=infocrit, njobs=njobs) 
