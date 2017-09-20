''' Python wrapper for P_l(k) calculation for the nonGaussLike project
'''
import subprocess
import os.path
import numpy as np 

import util as UT

def Catalog(catalog, n_mock):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if catalog == 'patchy': 
        return ''.join([UT.dat_dir(), 'patchy/Patchy-Mocks-DR12NGC-COMPSAM_V6C_', str("%04d" % n_mock), '.dat'])
    else:
        raise NotImplementedError


def Random(catalog):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if catalog == 'patchy': 
        return ''.join([UT.dat_dir(), 'Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x50.dat'])
    else:
        raise NotImplementedError


def FFT(catalog, Lbox=None, Ngrid=None, n_interp=None, P0=None, sys=None, comp=None, zbin=None): 
    ''' fft file name 
    '''
    if '.dat' in catalog: 
        pass 
    name = '/'.join(catalog.split('/')[:-1])+'/fft.'+catalog.split('/')[-1].split('.dat')[0]
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
    return name


def Plk(catalog, Lbox=None, Ngrid=None, n_interp=None, P0=None, sys=None, comp=None, zbin=None): 
    ''' P_l(k) for catalog 
    '''
    name = '/'.join(catalog.split('/')[:-1])+'/plk.'+catalog.split('/')[-1].split('.dat')[0]
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
    return name


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
    file_catalog = Catalog(catalog, n_mock)
    # zbin:               1 (z1), 2 (z2), 3 (z3)
    fft_exe = ''.join([UT.code_dir(), 'fort/FFT_scoccimarro_cmasslowzcomb.exe'])

    # construct mock FFT for z1, z2, z3 redshift bins 
    for zbin in [1,2,3]: 
        # fft name   
        file_fft = FFT(file_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        print 'Constructing FFT for ...'  
        print file_catalog 
        print file_fft
        print '' 
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
        print cmd_D
        subprocess.call(cmd_D.split())

    # construct random FFTs
    rand_catalog = random(catalog) 
    for zbin in [1,2,3]: 
        file_fft = FFT(rand_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
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
            subprocess.call(cmd_R.split())

    # construct P(k) using the catalog FFT and random FFT
    pk_exe = ''.join([UT.code_dir(), 'fort/power_scoccimarro.exe'])
    for zbin in [1,2,3]: 
        fft_D = FFT(file_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        fft_R = FFT(rand_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)

        file_plk = Plk(file_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        print 'Constructing ...'
        print file_plk

        cmd_plk = ' '.join([pk_exe,
            fft_D, 
            fft_R, 
            file_plk, 
            str(Lbox), 
            str(int(Ngrid/2))]) # N_bin = Ngrid/2 
        subprocess.call(cmd_plk.split())

    return None


if __name__=="__main__": 
    buildPk('patchy', 1, sys='fc')
