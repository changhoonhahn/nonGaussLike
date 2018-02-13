#!/bin/bash/
# e.g. python run_diverge.py (pk or gmf) (div_func) (Nref) (K) (n_mc) (n_jobs) 

for div in "renyi:.5" "kl"; do 
    for K in 10; do #5 8 10 15; do 
        for obv in "pk" #"gmf" #"pk"
        do 
            if [ $obv = "pk" ]; then 
                Nref=2000
            fi
            if [ $obv = "gmf" ]; then 
                Nref=10000
            fi
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div ref $Nref $K 100 -1
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pX_gauss $Nref $K 100 -1
            python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
                $obv $div pX_gauss_hartlap $Nref $K 100 -1
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pX_GMM $Nref $K 100 -1 30
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pX_GMM_ref $Nref $K 100 -1 30
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_scottKDE $Nref $K 100 -1
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pX_scottKDE_ref $Nref $K 100 -1
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_GMM $Nref $K 100 -1 30
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_GMM_ref $Nref $K 100 -1 30
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_parICA_GMM $Nref $K 100 -1 30
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_parICA_GMM_ref $Nref $K 100 -1 30

            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_ICA_scottKDE $Nref $K 100 6
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_ICA_scottKDE_ref $Nref $K 100 6
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_parICA_scottKDE $Nref $K 100 6
            #python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py \
            #    $obv $div pXi_parICA_scottKDE_ref $Nref $K 100 6
        done 
    done 
done 
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_KDE $Nref $K 100 6 30
#python /Users/chang/projects/nonGaussLike/nongausslike/run/run_diverge.py $obv $div pXi_ICA_KDE_ref $Nref $K 100 6 30
