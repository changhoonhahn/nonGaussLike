import numpy as np 
import emcee

# --- local ---
import util as UT
import model as Mod


def data(ell, zbin, nors): 
    ''' Read in P(k) measurements of data 
    '''
    if ell == 0:
        str_pole = 'mono'
    elif ell == 2:
        str_pole = 'quadru'
    elif ell == 4:
        str_pole = 'hexadeca'
    str_pole += 'pole'

    fname = ''.join([UT.dat_dir(), 'Beutler/public_material_RSD/',
        'Beutleretal_pk_', str_pole, '_DR12_', nors.upper(), '_z', str(zbin), '_prerecon_120.dat'])

    k_central, k_mean, pk = np.loadtxt(fname, skiprows=31, unpack=True, usecols=[0,1,2])
    return k_central, pk 


if __name__=="__main__":
    pass
