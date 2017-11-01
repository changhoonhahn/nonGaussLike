'''
evaluate model P(k) for Florian's MCMC chain 
'''
import sys 
import numpy as np 

import env 
import util as UT
import model as Mod
import infer as Inf


def Pk_model_mcmc(tag): 
    ''' Evalulate the P(k) model for the `tag` MCMC chain 
    '''
    if tag == 'beutler_z1': 
        # read in BOSS P(k) (read in data D) 
        k_list, pk_ngc_data = [], []
        for ell in [0,2,4]: 
            k, plk_ngc = Inf.data(ell, 1, 'ngc')
            k_list.append(k)
            pk_ngc_data.append(plk_ngc)
        binrange1, binrange2, binrange3 = len(k_list[0]), len(k_list[1]), len(k_list[2])
        maxbin1 = len(k_list[0])+1
        k = np.concatenate(k_list)

        # read in Florian's RSD MCMC chains  
        for ichain in range(4): # read in the 4 chains
            chain_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
                'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.dat']) 
            sample = np.loadtxt(chain_file, skiprows=1)
            chain = sample[:,1:]

            for i in range(sample.shape[0]):
                model_i = Mod.taruya_model(100, binrange1, binrange2, binrange3, maxbin1, k, 
                        chain[i,0], chain[i,1], chain[i,2], chain[i,3], chain[i,4], 
                        chain[i,5], chain[i,6], chain[i,7], chain[i,8], chain[i,9], chain[i,10])
                if i == 0: 
                    models_ngc = np.zeros((sample.shape[0], len(model_i[0])))
                    models_sgc = np.zeros((sample.shape[0], len(model_i[0])))
                models_ngc[i,:] = model_i[0]
                models_sgc[i,:] = model_i[1]

            chain_model_ngc_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
                'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.model.ngc.dat']) 
            np.savetxt(chain_model_ngc_file, models_ngc)

            chain_model_sgc_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
                'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.model.sgc.dat']) 
            np.savetxt(chain_model_sgc_file, models_sgc)
    else: 
        raise ValueError
    return None 

if __name__=="__main__": 
    tag = sys.argv[1]
    Pk_model_mcmc(tag)
