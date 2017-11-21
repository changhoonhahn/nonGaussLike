#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 03:00:00 
#SBATCH -J div_gmf_kde 
#SBATCH -o div_gmf_kde.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda

srun -n 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_diverge.py gmf kl pX_KDE 10000 10 400 20000 16
