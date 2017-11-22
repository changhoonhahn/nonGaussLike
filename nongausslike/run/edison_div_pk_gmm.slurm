#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 03:00:00 
#SBATCH -J div_pk_gmm_ica 
#SBATCH -o div_pk_gmm_ica.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
# input obvs div_func div Nref K n_mc ncomp_max

#srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_diverge.py pk kl pX_GMM 10000 10 1000 20 
srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_diverge.py pk kl pXi_ICA_GMM 10000 10 1000 20 
