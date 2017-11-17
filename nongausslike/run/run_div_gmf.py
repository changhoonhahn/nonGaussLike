import sys as Sys

import env 
import paper as Pap


if __name__=="__main__": 
    div_func = Sys.argv[1]
    Nref = Sys.argv[2]
    K = Sys.argv[3]
    n_mc = Sys.argv[4]
    density_method = Sys.argv[5]
    if density_method == 'gmm': 
        n_comp_max = Sys.argv[6]
        Pap.divGMF(div_func=div_func, Nref=Nref, K=K, n_mc=n_mc, 
                density_method=density_method, n_comp_max=n_comp_max)
    else: 
        Pap.divGMF(div_func=div_func, Nref=Nref, K=K, n_mc=n_mc, 
                density_method=density_method)
