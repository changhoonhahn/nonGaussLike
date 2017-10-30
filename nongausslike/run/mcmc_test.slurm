#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J mcmc_test 
#SBATCH -o mcmc_test.o%j 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 10 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run_mcmc.py testing 1 10 1 10
