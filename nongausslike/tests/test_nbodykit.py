from nbodykit.lab import *
from nbodykit import setup_logging, style
import numpy as np 
import os

import env
import util as UT 

def bosspk(zbin): 
    ''' calculate boss P(k) using nbody kit 
    ''' 
    path_to_catalogs = UT.catalog_dir('boss') 

    data = FITSCatalog(os.path.join(path_to_catalogs, 
        'galaxy_DR12v5_CMASSLOWZTOT_North.fits'))
    print('data columns = ', data.columns)
    randoms = FITSCatalog(os.path.join(path_to_catalogs, 
        'random1_DR12v5_CMASSLOWZTOT_North.fits'))
    print('randoms columns = ', randoms.columns)

    # the fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(H0=67.6, Om0=0.31, flat=True)
    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)
    randoms['WEIGHT'] = 1.0
    data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.0)
    
    if zbin == 1: 
        ZMIN, ZMAX = 0.2, 0.5
    elif zbin == 2:
        ZMIN, ZMAX = 0.4, 0.6

    randoms['Selection'] = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
    data['Selection'] = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)

    # combine the data and randoms into a single catalog
    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(Nmesh=256, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', 
            window='tsc')
    # compute the multipoles
    r = ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.01, kmin=0.)

    poles = r.poles
    for ell in [0, 2, 4]: 
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - r.attrs['shotnoise'] 
        np.savetxt(''.join([UT.catalog_dir('boss'), 'nbodykit.p', str(ell), 'k.dat']), 
                (np.vstack(np.array([poles['k'], P]))).T, 
                fmt=['%f', '%f'], delimiter='\t')
    return None


if __name__=="__main__": 
    bosspk(1)
