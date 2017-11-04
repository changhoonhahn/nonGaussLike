'''

Generate importance weights for different re-evaluations of the
likelihood. 

'''
import sys as Sys
import numpy as np 

import env 
import util as UT 
import infer as Inf


def importance_weight_Pk(tag_mcmc, tag_like, ichain): 
    ''' save importance weights to file 
    '''
    chain = Inf.mcmc_chains(tag_mcmc, ichain=ichain)
    ws = Inf.importance_weight(tag_like, chain, zbin=1) # watch out zbin hardcoded. 

    weight_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), 
        '.', tag_like, '_weights.dat']) 
    hdr = 'ln(P_denom), ln(P_nomin), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


def importance_weight_Gmf(tag_mcmc, tag_like, run): 
    ''' save importance weights to file 
    '''
    chain = Inf.mcmc_chains(tag_mcmc)
    ws = Inf.importance_weight(tag_like, chain, run=run) # watch out zbin hardcoded. 

    weight_file = ''.join([UT.dat_dir(), 'manodeep/', 
        'status_file_Consuelo_so_mvir_Mr19_box_4022_and_4002_fit_wp_0_fit_gmf_1_pca_0'
        '.run', str(run), '.', tag_like, '_weights.dat']) 

    hdr = 'ln(P_denom), ln(P_nomin), w_importance'
    np.savetxt(weight_file, np.array(ws).T, header=hdr) 
    return None 


if __name__=='__main__': 
    tag_mcmc = Sys.argv[1]
    tag_like = Sys.argv[2]
    if 'beutler' in tag_mcmc: 
        ichain = int(Sys.argv[3]) 
        importance_weight_Pk(tag_mcmc, tag_like, ichain) 
    elif 'manodeep' in tag_mcmc: 
        irun = int(Sys.argv[3]) 
        importance_weight_Gmf(tag_mcmc, tag_like, run=irun) 
