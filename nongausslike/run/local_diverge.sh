#!/bin/bash/
# e.g. python run_diverge.py (pk or gmf) (div_func) (Nref) (K) (n_mc) (n_jobs) 

#div="renyi:.5"
div="kl"
# --- P(k) --- 
#obv="pk"
#Nref=2000
# --- gmf --- 
obv="gmf"
Nref=10000


#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div ref $Nref 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_gauss $Nref 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_GMM $Nref 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_GMM_ref $Nref 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_KDE $Nref 10 100 -1
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_KDE_ref $Nref 10 100 -1
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_GMM $Nref 10 100 -1 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_GMM_ref $Nref 10 100 -1 30
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_KDE $Nref 10 100 6 30
python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_KDE_ref $Nref 10 100 6 30
