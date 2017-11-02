#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J pk_boss 
#SBATCH -o pk_boss.o 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_pk_boss.py 1
