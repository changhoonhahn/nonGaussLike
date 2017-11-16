'''

Generate importance weights for different re-evaluations of the
likelihood. 

'''
import sys as Sys
import numpy as np 

import env 
import util as UT 
import infer as Inf


def importance_weight_Pk(tag_like, ichain, zbin=1): 
    ''' save importance weights to file 
    '''
    chain = Inf.mcmc_chains('beutler_z'+str(zbin), ichain=ichain)
    ws = Inf.W_importance(tag_like, chain, zbin=zbin) # watch out zbin hardcoded. 

    weight_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z', str(zbin), '_chain', str(ichain), 
        '.', tag_like, '_weights.dat']) 
    hdr = 'ln(P_denom), ln(P_numer), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


def importance_weight_Gmf(tag_like): 
    ''' save importance weights for Manodeep's MCMC
    chain to file 
    '''
    chain = Inf.mcmc_chains('manodeep')
    ws = Inf.W_importance(tag_like, chain)

    weight_file = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0',
        '.', tag_like, '_weights.dat']) 

    hdr = 'ln(P_denom), ln(P_numer), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


if __name__=='__main__': 
    # e.g. python run/run_importance_weight.py gmf_pca_chi2 manodeep
    tag_like = Sys.argv[1]
    tag_mcmc = Sys.argv[2]
    if 'beutler' in tag_mcmc: 
        ichain = int(Sys.argv[3]) 
        importance_weight_Pk(tag_like, ichain) 
    elif 'manodeep' in tag_mcmc: 
        importance_weight_Gmf(tag_like) 
