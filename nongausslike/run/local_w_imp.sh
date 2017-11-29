#!/bin/bash/

# P(k) L = PI p_KDE(Xi parallel ICA) 
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_importance_weight.py \
#    pk RSD_pXiICA_gauss parallel kde 4
# P(k) L = PI p_KDE(Xi deflation ICA) 
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_importance_weight.py \
    pk RSD_pXiICA_gauss deflation kde 4

#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_importance_weight.py gmf_pX_chi2 manodeep gmm 30 bic 
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_importance_weight.py gmf_pXiICA_chi2 manodeep kde 4
