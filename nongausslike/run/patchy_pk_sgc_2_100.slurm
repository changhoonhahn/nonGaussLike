#!/bin/bash -l 
#SBATCH -p shared 
#SBATCH -N 1 
#SBATCH -t 06:00:00 
#SBATCH -J patchy_pk_sgc_2_100
#SBATCH -o patchy_pk_sgc_2_100.o%j 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/pk.py 2 100 sgc 
