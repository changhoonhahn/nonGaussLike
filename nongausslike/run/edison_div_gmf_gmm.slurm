#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 03:00:00 
#SBATCH -J div_gmf_gmm 
#SBATCH -o div_gmf_gmm.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_diverge.py gmf kl pX_GMM 10000 10 1000 20 
