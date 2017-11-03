'''

Generate importance weights for different re-evaluations of the
likelihood. 

'''
import sys as Sys
import numpy as np 

import env 
import util as UT 
import infer as Inf


def importance_weight(tag_mcmc, tag_like, ichain): 
    ''' save importance weights to file 
    '''
    chain = Inf.mcmc_chains(tag_mcmc, ichain)
    ws = Inf.importance_weight(tag_like, chain, zbin=1) # watch out zbin hardcoded. 

    weight_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
        'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), 
        '.', tag_like, '_weights.dat']) 
    hdr = 'P_denom, P_nomin, w_importance'
    np.savetxt(weight_file, ws.T, header=hdr) 
    return None 


if __name__=='__main__': 
    tag_mcmc = Sys.argv[1]
    tag_like = Sys.argv[2]
    ichain = int(Sys.argv[3]) 

    importance_weight(tag_mcmc, tag_like, ichain) 
