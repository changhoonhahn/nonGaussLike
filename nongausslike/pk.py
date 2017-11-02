''' Python wrapper for P_l(k) calculation for the nonGaussLike project
'''
from nbodykit.lab import *
import subprocess
import os.path
import numpy as np 

import util as UT
from interruptible_pool import InterruptiblePool as Pewl


def Catalog(catalog, n_mock, NorS='ngc'):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if 'patchy' in catalog: 
        return ''.join([UT.catalog_dir('patchy'), 'Patchy-Mocks-DR12', NorS.upper(), '-COMPSAM_V6C_', str("%04d" % n_mock), '.dat'])
    elif catalog == 'boss': 
        if NorS == 'ngc': 
            return ''.join([UT.catalog_dir('boss'), 'galaxy_DR12v5_CMASSLOWZTOT_North.dat']) 
    else:
        raise NotImplementedError


def Random(catalog, NorS='ngc'):
    ''' name of the original catalog files. These will be fed into the FFT 
    '''
    if catalog == 'patchy': 
        return ''.join([UT.dat_dir(), 'patchy/Patchy-Mocks-Randoms-DR12', NorS.upper(), '-COMPSAM_V6C_x50.dat'])
    elif catalog == 'boss': 
        if NorS == 'ngc': 
            return ''.join([UT.catalog_dir('boss'), 'random1_DR12v5_CMASSLOWZTOT_North.dat'])
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


def Plk(catalog, Lbox=None, Ngrid=None, n_interp=None, P0=None, sys=None, comp=None, Nbin=None, zbin=None): 
    ''' P_l(k) for catalog 
    '''
    name = '/'.join(catalog.split('/')[:-1])+'/plk.'+catalog.split('/')[-1].split('.dat')[0]
    name += '.Lbox'+str(Lbox)
    name += '.Ngrid'+str(Ngrid)
    name += '.Nbin'+str(Nbin)
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
    if catalog == 'boss': 
        idata = 1
        Lboxs = [2800, 3200, 3800]     # Box size 
        Ngrids = [360, 410, 490]     # FFT grid size (480 or 960)
        Nbins = [40, 40, 40]
        comp_flag = 0
    elif catalog == 'patchy': # boss or patchy:     1 or 2 
        idata = 2
        Lboxs = [2800, 3600, 3600]     # Box size 
        Ngrids = [360, 480, 480]     # FFT grid size (480 or 960)
        Nbins = [40, 40, 40]
        comp_flag = 1
    else: 
        raise ValueError
    n_interp = 4    # interpolation (4 or 2) 
    P0 = 10000      # P0:                 10000
    if sys == 'fc': # fc flag: 1 for fc 0 for no fc  
        fc_flag = 1 
    else: 
        fc_flag = 0 
    file_catalog = Catalog(catalog, n_mock) # catalog file 
    rand_catalog = Random(catalog) # random file 

    # construct FFTs for mock and random, then P_l(k)
    # for z1, z2, z3 redshift bins 
    fft_exe = ''.join([UT.code_dir(), 'fort/FFT_scoccimarro_cmasslowzcomb.exe'])
    pk_exe = ''.join([UT.code_dir(), 'fort/power_scoccimarro.exe'])
    for iz, zbin in enumerate([1]): #,2,3]): only 1 for now 
        # data fft name   
        fft_D = FFT(file_catalog, Lbox=Lboxs[iz], Ngrid=Ngrids[iz], n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        # random fft name   
        fft_R = FFT(rand_catalog, Lbox=Lboxs[iz], Ngrid=Ngrids[iz], n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)
        # pk name   
        file_plk = Plk(file_catalog, Lbox=Lboxs[iz], Ngrid=Ngrids[iz], Nbin=Nbins[iz], n_interp=n_interp, P0=P0, 
                sys=sys, comp=comp_flag, zbin=zbin)

        if not os.path.isfile(fft_D): 
            print('Constructing FFT for ...')
            print(file_catalog)
            print(fft_D)
            print('')
            cmd_D = ' '.join([fft_exe, 
                str(idata), 
                str(Lboxs[iz]), 
                str(Ngrids[iz]), 
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
        if not os.path.isfile(fft_R): 
            cmd_R = ' '.join([fft_exe, 
                str(idata), 
                str(Lboxs[iz]), 
                str(Ngrids[iz]), 
                str(n_interp), 
                str(1), # random 
                str(P0), 
                str(fc_flag), 
                str(comp_flag), 
                rand_catalog, 
                str(zbin), 
                fft_R])
            print('Constructing random FFT ...')
            print(rand_catalog)
            print(fft_R)
            print('')
            subprocess.call(cmd_R.split())

        # construct P(k) using the catalog FFT and random FFT
        if not os.path.isfile(file_plk): 
            print('Constructing ...')
            print(file_plk)
            cmd_plk = ' '.join([pk_exe,
                fft_D, 
                fft_R, 
                file_plk, 
                str(Lboxs[iz]), 
                str(Nbins[iz])]) 
            subprocess.call(cmd_plk.split())
    return None


def boss_preprocess(NorS='ngc'): 
    ''' read in and pre-process boss data 
    '''
    if NorS == 'ngc': 
        str_NorS = 'North'
    elif NorS == 'sgc': 
        str_NorS = 'South'

    # read in original data in fits file format 
    f_orig = ''.join([UT.catalog_dir('boss'), 'galaxy_DR12v5_CMASSLOWZTOT_', str_NorS, '.fits']) 
    data = mrdfits(f_orig)
    # data columns: ra,dec,az,nbb,wsys,wnoz,wcp,comp
    data_list = [data.ra, data.dec, data.z, data.nz, data.weight_systot, data.weight_noz, data.weight_cp, data.comp]
    data_fmt = ['%f', '%f', '%f', '%e', '%f', '%f', '%f', '%f']
    header_str = "columns: ra,dec,az,nbb,wsys,wnoz,wcp,comp" 

    f_boss = ''.join([UT.catalog_dir('boss'), 'galaxy_DR12v5_CMASSLOWZTOT_North.dat']) 
    np.savetxt(f_boss,
            (np.vstack(np.array(data_list))).T, 
            fmt=data_fmt, 
            delimiter='\t', 
            header=header_str) 

    # now random catalog 
    f_orig = ''.join([UT.catalog_dir('boss'), 'random1_DR12v5_CMASSLOWZTOT_', str_NorS, '.fits.gz']) 
    #data = mrdfits(f_orig)
    # data columns: ra,dec,az,nbb,wsys,wnoz,wcp,comp
    #data_list = [data.ra, data.dec, data.z, data.nz, data.weight_systot, data.weight_noz, data.weight_cp, data.comp]

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
    f.close() 
    return None


def Pk_NBKT_boss(zbin, NorS='ngc'): 
    ''' calculate boss P(k) using nbodykit, which uses Nick's estimator 
    ''' 
    # first read in data and random catalogs 
    path_to_catalogs = UT.catalog_dir('boss') 
    if NorS == 'ngc': str_nors = 'North'
    else: str_nors = 'South'
    data = FITSCatalog(os.path.join(path_to_catalogs, 
        'galaxy_DR12v5_CMASSLOWZTOT_'+str_nors+'.fits'))
    randoms = FITSCatalog(os.path.join(path_to_catalogs, 
        'random1_DR12v5_CMASSLOWZTOT_'+str_nors+'.fits'))
    
    # impose fiducial BOSS DR12 cosmology
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
    mesh = fkp.to_mesh(Nmesh=360, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')
    # compute the multipoles
    r = ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.01, kmin=0.)

    poles = r.poles
    plk = [] 
    for ell in [0, 2, 4]: 
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - r.attrs['shotnoise'] 
        plk.append(P)

    # shot noise comparison 
    #print(r.attrs['shotnoise'])
    #alpha = r.attrs['alpha']
    #sel = mesh['randoms'][mesh.selection]
    #first = mesh['randoms'][sel]
    #comp_weight = first[mesh.comp_weight]
    #fkp_weight = first[mesh.fkp_weight]
    #S = (comp_weight**2*fkp_weight**2).sum()
    #print("expected SN")
    #print((1.+alpha)*alpha*S/r.attrs['randoms.norm'])
    #sel = mesh['data'][mesh.selection]
    #first = mesh['data'][sel]
    #comp_weight = first[mesh.comp_weight]
    #fkp_weight = first[mesh.fkp_weight]
    #S_data = (comp_weight**2*fkp_weight**2).sum()
    #print("true SN")
    #print((S_data+alpha**2*S)/r.attrs['randoms.norm'])

    f = open(''.join([UT.catalog_dir('boss'), 'pk.nbodykit.', NorS, '.zbin', str(zbin), '.dat']), 'w')
    f.write("### header ### \n")
    for key in r.attrs:
        f.write("%s = %s \n" % (key, str(r.attrs[key])))
    f.write("columns : k , P0, P2, P4 \n")
    f.write('### header ### \n') 

    for ik in range(len(poles['k'])): 
        f.write("%f \t %f \t %f \t %f" % (poles['k'][ik], plk[0][ik], plk[1][ik], plk[2][ik]))
        f.write("\n") 
    f.close() 
    return None


def Pk_NBKT_patchy(i_mock, zbin, NorS, randoms=None): 
    ''' calculate Patchy mock P(k) using nbodykit, which uses Nick's estimator 
    ''' 
    plk_name = ''.join([UT.catalog_dir('patchy'), 'pk.patchy.', str(i_mock), '.', NorS, '.zbin', str(zbin), '.nbodykit.dat'])
    if os.path.isfile(plk_name):
        return None 
    print('patchy --',i_mock)
    # first read in data and random catalogs 
    col_data = ['RA', 'DEC', 'Z', 'DUM0', 'NZ', 'DUM1', 'VETO', 'WRED']
    data = CSVCatalog(Catalog('patchy', i_mock, NorS=NorS), col_data)
    if randoms is None: 
        col_random = ['RA', 'DEC', 'Z', 'NZ', 'BIAS', 'VETO', 'WRED']
        randoms = CSVCatalog(Random('patchy', NorS=NorS), col_random)
    
    # impose fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(H0=67.6, Om0=0.31, flat=True)
    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)
    randoms['WEIGHT'] = randoms['WRED']
    randoms['WEIGHT_FKP'] = 1./(1.+ randoms['NZ'] * 1e4)
    data['WEIGHT'] = data['WRED']
    data['WEIGHT_FKP'] = 1./(1.+ data['NZ'] * 1e4)
    
    if zbin == 1: 
        ZMIN, ZMAX = 0.2, 0.5
    elif zbin == 2:
        ZMIN, ZMAX = 0.4, 0.6
    elif zbin == 3: 
        ZMIN, ZMAX = 0.5, 0.75

    randoms['Selection'] = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)&(randoms['VETO'] > 0)
    data['Selection'] = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)&(data['VETO'] > 0)

    # combine the data and randoms into a single catalog
    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(Nmesh=360, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WRED', window='tsc')
    # compute the multipoles
    r = ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.01, kmin=0.)

    poles = r.poles
    plk = [] 
    for ell in [0, 2, 4]: 
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - r.attrs['shotnoise'] 
        plk.append(P)
    f = open(plk_name, 'w')
    f.write("### header ### \n")
    for key in r.attrs:
        f.write("%s = %s \n" % (key, str(r.attrs[key])))
    f.write("columns : k , P0, P2, P4 \n")
    f.write('### header ### \n') 

    for ik in range(len(poles['k'])): 
        f.write("%f \t %f \t %f \t %f" % (poles['k'][ik], plk[0][ik], plk[1][ik], plk[2][ik]))
        f.write("\n") 
    f.close() 
    print('wrote:', plk_name)
    return None


def Pk_NBKT_patchy_wrap(nmock0, nmock1, zbin, NorS='ngc'): 
    '''
    '''
    col_random = ['RA', 'DEC', 'Z', 'NZ', 'BIAS', 'VETO', 'WRED']
    randoms = CSVCatalog(Random('patchy', NorS=NorS), col_random)

    for i_mock in range(nmock0, nmock1+1): 
        Pk_NBKT_patchy(i_mock, zbin, NorS, randoms=randoms)
    return None 


def _make_run_nbkt(NorS): 
    '''
    '''
    n0 = np.arange(100, 2100, 100)+1
    #n0[0] = 2
    n1 = np.arange(200, 2148, 100)
    n1[-1] = 2048
    for i in range(len(n0)): 
        run_file = ''.join([UT.run_dir(), 'patchy_nbkt_pk_', NorS, '_', str(n0[i]), '_', str(n1[i]), '.slurm'])
        f = open(run_file, 'w')

        f.write('#!/bin/bash -l \n')
        f.write('#SBATCH -p regular \n') 
        f.write('#SBATCH -N 1 \n') 
        f.write('#SBATCH -t 06:00:00 \n')
        f.write('#SBATCH -J patchy_nbkt_pk_'+NorS+'_'+str(n0[i])+'_'+str(n1[i])+' \n')
        f.write('#SBATCH -o patchy_nbkt_pk_'+NorS+'_'+str(n0[i])+'_'+str(n1[i])+'.o%j \n')
        f.write('#SBATCH -L SCRATCH,project \n')
        f.write('\n')
        f.write('source /usr/common/contrib/bccp/conda-activate.sh 2.7 \n') 
        f.write('\n')
        f.write('srun -n 1 -c 1 python /global/homes/c/chahah/projects/nonGaussLike/nongausslike/run/run_pk.py '+str(n0[i])+' '+str(n1[i])+' '+NorS)
    f.close() 
    return None


if __name__=="__main__": 
    _make_run_nbkt('sgc')
