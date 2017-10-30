#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 06:00:00 
#SBATCH -J patchy_nbkt_pk_1601_1700 
#SBATCH -o patchy_nbkt_pk_1601_1700.o%j 
#SBATCH -L SCRATCH,project 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/pk.py 1 1601 1700