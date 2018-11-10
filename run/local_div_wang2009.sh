#!/bin/bash/
# calculate the following KL divergences using the Wang et al. (2009) reduced bias estimator 
# - KL( X || N(mu_X, C_X) ) 
# - KL( X_gauss || N(mu_X, C_X) )  reference distribution 
# for P(k) and GMF analyses 
obvs='gmf' #'pk'

for Nref in 1000 2000 4000 8000 10000; do 
    python $HOME/projects/nonGaussLike/run/run_div_wang2009.py $obvs "pX_gauss" "XY" $Nref 100 
    python $HOME/projects/nonGaussLike/run/run_div_wang2009.py $obvs "ref" "XY" $Nref 100 
done 
