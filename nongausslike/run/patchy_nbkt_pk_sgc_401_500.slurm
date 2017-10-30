#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 06:00:00 
#SBATCH -J patchy_nbkt_pk_sgc_401_500 
#SBATCH -o patchy_nbkt_pk_sgc_401_500.o%j 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 2.7 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_pk.py 401 500 sgc