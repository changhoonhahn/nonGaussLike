import numpy as np 
import emcee
from emcee.utils import MPIPool

# --- local ---
import util as UT
import data as Dat 
import model as Mod
import nongauss as NG 


def importance_sampling(mcmc_tag, like_ratio):
    ''' Use importance sampling in order to estimate the new posterior 
    distribution. A mcmc chain is read in then weighted by the likelihood
    ratio. 
    '''
    chain = mcmc_chains(mcmc_tag)
    ws = like_ratio(chain)


def likeRatio(chain, tag, **kwargs): 
    ''' Given a dictionary with the MCMC chain, evaluate the likelihood ratio 
    '''
    if 'RSD' in tag: # Florian's RSD analysis 
        if 'zbin' not in kwargs.keys(): 
            raise ValueError('specify zbin in kwargs') 

        # read in BOSS P(k) (read in data D) 
        k_list, pk_ngc_data, pk_sgc_data = [], [], []
        for ell in [0,2,4]: 
            k, plk_ngc = data(ell, kwargs['zbin'], 'ngc')
            _, plk_sgc = data(ell, kwargs['zbin'], 'sgc')
            k_list.append(k)
            pk_ngc_data.append(plk_ngc)
            pk_sgc_data.append(plk_sgc)

        binrange1, binrange2, binrange3 = len(k_list[0]), len(k_list[1]), len(k_list[2])
        maxbin1 = len(k_list[0])+1
        k = np.concatenate(k_list)
        
        # calculate D - m(theta) for all the mcmc chain
        delta_ngc, delta_sgc = [], [] 
        for i in range(len(chain['chi2'])): 
            model_i = Mod.taruya_model(100, binrange1, binrange2, binrange3, maxbin1, k, 
                    chain['alpha_perp'][i], 
                    chain['alpha_para'][i], 
                    chain['fsig8'][i], 
                    chain['b1sig8_NGC'][i], 
                    chain['b1sig8_SGC'][i], 
                    chain['b2sig8_NGC'][i], 
                    chain['b2sig8_SGC'][i], 
                    chain['N_NGC'][i], 
                    chain['N_SGC'][i], 
                    chain['sigmav_NGC'][i], 
                    chain['sigmav_SGC'][i])

            delta_ngc.append(model[0] - np.concatenate(pk_ngc_data))
            delta_sgc.append(model[0] - np.concatenate(pk_sgc_data))

        # import PATCHY mocks 
        pk_ngc_list, pk_sgc_list = [], [] 
        for ell in [0, 2, 4]:
            pk_ngc_list.append(NG.dataX('patchy.ngc.z'+str(zbin), ell=ell, sys='fc'))
            pk_sgc_list.append(NG.dataX('patchy.sgc.z'+str(zbin), ell=ell, sys='fc'))
        pk_ngc_mock = np.concatenate(pk_ngc_list) 
        pk_sgc_mock = np.concatenate(pk_sgc_list) 

    if tag == 'RSD_ica_pca': # P_ICA(D - m(theta)) / P_PCA(D - m(theta))
        lnP_ica_ngc = NG.lnL_ica(np.array(delta_ngc), pk_ngc_mock) 
        lnP_pca_ngc = NG.lnL_pca(np.array(delta_ngc), pk_ngc_mock) 
        
        lnP_ica_sgc = NG.lnL_ica(np.array(delta_sgc), pk_sgc_mock) 
        lnP_pca_sgc = NG.lnL_pca(np.array(delta_sgc), pk_sgc_mock) 

    


def mcmc_chains(tag): 
    ''' Given some tag string return mcmc chain in a dictionary 
    '''
    if tag == 'beutler_z1': 
        # read in Florian's RSD MCMC chains  
        for ichain in range(4): # read in the 4 chains
            chain_file = ''.join([UT.dat_dir(), 'Beutler/public_full_shape/', 
                'Beutler_et_al_full_shape_analysis_z1_chain', str(ichain), '.dat']) 
            sample = np.loadtxt(chain_file, skiprows=1)
            if ichain == 0: 
                chain = sample[:,1:]
            else: 
                chain = np.concatenate([chain, sample[:,1:]])
        labels = ['alpha_perp', 'alpha_para', 'fsig8', 'b1sig8_NGC', 'b1sig8_SGC', 'b2sig8_NGC', 'b2sig8_SGC', 
                'N_NGC', 'N_SGC', 'sigmav_NGC', 'sigmav_SGC', 'chi2'] 
        chain_dict = {} 
        for i in range(len(labels)): 
            chain_dict[labels[i]] = chain[:,i]
    else: 
        raise ValueError
    return chain_dict 


if __name__=="__main__": 
    mcmc_chains('beutler_z1')
