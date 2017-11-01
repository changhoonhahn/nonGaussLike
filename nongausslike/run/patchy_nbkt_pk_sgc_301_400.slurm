#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 02:00:00 
#SBATCH -J patchy_nbkt_pk_sgc_301_400 
#SBATCH -o patchy_nbkt_pk_sgc_301_400.o 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_pk.py 301 400 sgc
