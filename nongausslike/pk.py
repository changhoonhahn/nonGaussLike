''' Python wrapper for P_l(k) calculation for the nonGaussLike project
'''
import subprocess
import os.path
import numpy as np 
import sys as Sys

import util as UT
from interruptible_pool import InterruptiblePool as Pewl


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
        return ''.join([UT.dat_dir(), 'patchy/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x50.dat'])
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
    Ngrid = 480     # FFT grid size (480 or 960)
    n_interp = 4    # interpolation (4 or 2) 
    P0 = 10000      # P0:                 10000
    if sys == 'fc': # fc flag: 1 for fc 0 for no fc  
        fc_flag = 1 
    else: 
        fc_flag = 0 
    if catalog == 'patchy': # comp flag: 1 for comp = 1
        comp_flag = 1
    file_catalog = Catalog(catalog, n_mock) # catalog file 
    rand_catalog = Random(catalog) # random file 

    # construct FFTs for mock and random, then P_l(k)
    # for z1, z2, z3 redshift bins 
    fft_exe = ''.join([UT.code_dir(), 'fort/FFT_scoccimarro_cmasslowzcomb.exe'])
    pk_exe = ''.join([UT.code_dir(), 'fort/power_scoccimarro.exe'])
    for zbin in [1,2,3]: 
        # fft name   
        fft_D = FFT(file_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        print 'Constructing FFT for ...'  
        print file_catalog 
        print fft_D 
        print '' 
        cmd_D = ' '.join([fft_exe, 
            str(idata), 
            str(Lbox), 
            str(Ngrid), 
            str(n_interp), 
            str(0), # data 
            str(P0), 
            str(fc_flag), 
            str(comp_flag), 
            file_catalog, 
            str(zbin), 
            fft_D])
        subprocess.call(cmd_D.split())

        # construct random FFTs
        fft_R = FFT(rand_catalog, Lbox=Lbox, Ngrid=Ngrid, n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        if not os.path.isfile(fft_R): 
            cmd_R = ' '.join([fft_exe, 
                str(idata), 
                str(Lbox), 
                str(Ngrid), 
                str(n_interp), 
                str(1), # random 
                str(P0), 
                str(fc_flag), 
                str(comp_flag), 
                rand_catalog, 
                str(zbin), 
                fft_R])
            print 'Constructing random FFT ...'  
            print rand_catalog 
            print fft_R 
            print '' 
            subprocess.call(cmd_R.split())
        else: 
            print 'Random FFT ...'  
            print fft_R 
            print '' 

        # construct P(k) using the catalog FFT and random FFT
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


def buildPk_wrap(arg): 
    n_mock = arg[0]
    buildPk('patchy', n_mock, sys='fc')
    return None 


def _make_run(): 
    '''
    '''
    n0 = np.arange(0, 2100, 100)+1
    n0[0] = 2
    n1 = np.arange(100, 2148, 100)
    n1[-1] = 2048
    for i in range(len(n0)): 
        run_file = ''.join([UT.run_dir(), 'patchy_pk_', str(n0[i]), '_', str(n1[i])])
        f = open(run_file, 'w')

        f.write('#!/bin/bash -l \n')
        f.write('#SBATCH -p regular \n') 
        f.write('#SBATCH -N 1 \n') 
        f.write('#SBATCH -t 05:00:00 \n')
        f.write('#SBATCH -J patchy_pk_'+str(n0[i])+'_'+str(n1[i])+' \n')
        f.write('#SBATCH -o patchy_pk_'+str(n0[i])+'_'+str(n1[i])+'.o%j \n')
        f.write('#SBATCH -L SCRATCH,project \n')
        f.write('\n')
        f.write('module load python/2.7-anaconda \n') 
        f.write('\n')
        f.write('srun -n 1 -c 5 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/pk.py 5 '+str(n0[i])+' '+str(n1[i]))
    return None


if __name__=="__main__": 
    Nthreads = int(Sys.argv[1])
    print 'running on ', Nthreads, ' threads'
    pool = Pewl(processes=Nthreads)
    mapfn = pool.map

    nmock0 = int(Sys.argv[2])
    nmock1 = int(Sys.argv[3])
    arglist = [[i_mock] for i_mock in range(nmock0, nmock1+1)]

    mapfn(buildPk_wrap, [arg for arg in arglist])
    pool.close()
    pool.terminate()
    pool.join() 
