#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 02:00:00 
#SBATCH -J patchy_nbkt_pk_sgc_1301_1400 
#SBATCH -o patchy_nbkt_pk_sgc_1301_1400.o 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_pk.py 1301 1400 sgc
