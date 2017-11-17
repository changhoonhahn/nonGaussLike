#!/bin/bash -l 
#SBATCH -p debug 
#SBATCH -N 1 
#SBATCH -t 00:10:00 
#SBATCH -J div_gmf 
#SBATCH -o div_gmf.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda

srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_div_gmf.py kl 10000 10 10 gkde
