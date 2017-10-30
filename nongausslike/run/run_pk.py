import sys as Sys
import env 

from pk import Pk_NBKT_patchy_wrap
    
    
nmock0 = int(Sys.argv[1])
nmock1 = int(Sys.argv[2])
NorS = Sys.argv[3]

if NorS not in ['ngc', 'sgc']: 
    raise ValueError

Pk_NBKT_patchy_wrap(nmock0, nmock1, 1, NorS=NorS)
