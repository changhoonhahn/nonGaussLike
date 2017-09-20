''' Python wrapper for P_l(k) calculation for the nonGaussLike project
'''
import subprocess
import os.path
import numpy as np 


def catalog(catalog, n_mock):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if catalog == 'patchy': 
        return ''.join([UT.dat_dir(), 'patchy/Patchy-Mocks-DR12NGC-COMPSAM_V6C_', str("%04d" % n_mock), '.dat']])
    else:
        raise NotImplementedError


def random(catalog):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if catalog == 'patchy': 
        return ''.join([UT.dat_dir(), 'Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x50.dat'])
    else:
        raise NotImplementedError


def fft(catalog_name, Lbox=None, Ngrid=None, n_interp=None, P0=None, sys=None, comp=None, zbin=None): 
    ''' fft file name 
    '''
    if catalog == 'patchy': 
        name = catalog_name.split('.dat')[0]
        name += '.Lbox'+str(Lbox)
        name += '.Ngrid'+str(Ngrid)
        if n_interp == 2: 
            name += '.CICintp'
        else: 
            name += '.O4intp'
        name += '.P0'+str(P0)
        if sys == 'fc': 
            name += '.fc'
        if comp != 1: 
            name += '.comp'
        name += '.z'+str(zbin)
    else:
        raise NotImplementedError
    return Name


def buildPk(catalog, n_mock, sys=None): 
    ''' Calculate the powerspectrum multipoles for 
    specified catalog 
    '''
    if catalog == 'patchy': # boss or patchy:     1 or 2 
        idata = 2
    else: 
        raise ValueError
    Lbox = 3600     # Box size 
    Ngrid = 480     # FFT grid size:      480/960
    n_interp = 4    # interpolation:      4 (or 2) 
    # mock or random:     0 or 1  
    P0 = 10000      # P0:                 10000
    if sys == 'fc': # fc flag: 1 for fc 0 for no fc  
        fc_flag = 1 
    else: 
        fc_flag = 0 
    if catalog == 'patchy': # comp flag: 1 for comp = 1
        comp_flag = 1
    # catalog name 
    file_catalog = catalog(catalog, n_mock)
    # zbin:               1 (z1), 2 (z2), 3 (z3)
    # fft name   
    file_fft = fft(file_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
            sys=sys, comp=comp_flag, zbin=zbin)
    
    fft_exe = ''.join([UT.code_dir(), 'exe/FFT_scoccimarro_cmasslowzcomb.exe'])
    
    # construct mock FFT for z1, z2, z3 redshift bins 
    for zbin in [1,2,3]: 
        cmd_D = ' '.join([fft_exe, 
            str(idata), 
            str(Lbox), 
            str(Ngrid), 
            str(n_interp), 
            str(0),
            str(P0), 
            str(fc_flag), 
            str(comp_flag), 
            file_catalog, 
            str(zbin), 
            file_fft])
        subprocess.call(cmd_D.split())

    # construct random FFTs
    rand_catalog = random(catalog) 
    for zbin in [1,2,3]: 
        file_fft = fft(rand_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        if not os.path.isfile(file_fft): 
            cmd_R = ' '.join([fft_exe, 
                str(idata), 
                str(Lbox), 
                str(Ngrid), 
                str(n_interp), 
                str(1),
                str(P0), 
                str(fc_flag), 
                str(comp_flag), 
                file_catalog, 
                str(zbin), 
                file_fft])
            subprocess.call(cmd_D.split())
    
    # construct P(k) 
    
