import sys 
from infer import mcmc 

# python run_mcmc.py tag zbin nwalkers nchains minlength

# tag
tag = sys.argv[1]
print('running ', tag, ' MCMC')

# zbin 
zbin = int(sys.argv[2]) 
print('redshift bin ', str(zbin))

# walkers
nwalkers = int(sys.argv[3])
print(str(nwalkers), ' walkers')

# number of independent chains
nchains = int(sys.argv[4])
print(str(nchains), ' chains')

# min iteration length 
niter_min = int(sys.argv[5])

mcmc(tag=tag, zbin=zbin, nwalkers=nwalkers, Nchains=nchains, minlength=niter_min)
