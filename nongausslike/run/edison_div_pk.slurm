#!/bin/bash -l 
#SBATCH -p debug 
#SBATCH -N 1 
#SBATCH -t 00:10:00 
#SBATCH -J div_gmf_gmm 
#SBATCH -o div_gmf_gmm.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_diverge.py gmf kl pX_GMM 1000 10 10 20 
