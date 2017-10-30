#!/bin/bash -l 
#SBATCH -p shared 
#SBATCH -N 1 
#SBATCH -t 00:10:00 
#SBATCH -J patchy_pk_sgc_test 
#SBATCH -o patchy_pk_sgc_test.o%j 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 7 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/pk.py 101 101 sgc 
