import sys as Sys
import env 

from pk import Pk_NBKT_boss
    
zbin = int(Sys.argv[1])

for NorS in ['ngc', 'sgc']: 
    Pk_NBKT_boss(zbin, NorS=NorS)
