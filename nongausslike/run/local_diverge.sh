#!/bin/bash/
# e.g. python run_diverge.py (pk or gmf) (div_func) (Nref) (K) (n_mc) (n_jobs) 

#div="renyi:.5"
div="kl"

# --- P(k) --- 
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div ref 2000 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pX_gauss 2000 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pX_GMM 2000 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pX_GMM_ref 2000 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pX_KDE 2000 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pX_KDE_ref 2000 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pXi_ICA_GMM 2000 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pXi_ICA_GMM_ref 2000 10 100 -1 30
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pXi_ICA_KDE 2000 10 100 6 30
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py pk $div pXi_ICA_KDE_ref 2000 10 100 6 30

# --- gmf --- 
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py gmf $div ref 10000 10 100 -1 20000
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py gmf $div pX_gauss 10000 10 100 -1 20000
