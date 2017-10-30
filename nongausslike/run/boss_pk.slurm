#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 00:20:00 
#SBATCH -J boss_pk_nbodykit
#SBATCH -o boss_pk_nbodykit.o%j 

source /usr/common/contrib/bccp/conda-activate.sh 3.6 

# run the main nbodykit example
srun -n 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/pk.py
