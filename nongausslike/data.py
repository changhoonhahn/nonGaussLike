'''

code for assessing galaxy clustering data and etc. 


'''
import os 
import numpy as np 
import tarfile 
# -- local -- 
import util as UT


class Gmf: 
    def __init__(self):
        ''' GMF object
        '''
        self.nbin_low = None 
        self.nbin_high = None 
        self.nbins = None 
        self.gmf = None 

    def Observation(self): 
        ''' Read in gmf(N) measurements of SDSS Mr < -19 from Manodeep  
        '''
        f = ''.join([UT.dat_dir(), 'manodeep/sdss_Mr19_fib0_gmf_northonly.dat'])
        nbin_low, nbin_high, gmf = np.loadtxt(f, unpack=True, usecols=[0,1,2])
        nbins = np.concatenate([nbin_low, [nbin_high[-1]]])

        self.nbin_low_obv = nbin_low
        self.nbin_high_obv = nbin_high
        self.nbin_obv = nbins
        self.gmf_obv = gmf
        return nbins, gmf 

    def Read(self, name, ireal, ibox): 
        ''' Read in ireal-th realization and ibox-th box mock catalog 
        from run-th run. 
        '''
        f = ''.join([UT.catalog_dir(name), self._file_name(name, ireal, ibox)]) 
        gmf, nbin_low, nbin_high = np.loadtxt(f, unpack=True, usecols=[0,1,2])

        self.gmf = gmf
        self.nbin_low = nbin_low 
        self.nbin_high = nbin_high 
        self.nbins = np.concatenate([nbin_low, [nbin_high[-1]]])
        return None

    def _n_mock(self, name): 
        ''' Given name of mock return n_mock. 
        '''
        if 'manodeep' in name:
            n_mock = 200 
        else: 
            raise ValueError
        return n_mock 
    
    def _file_name(self, name, ireal, ibox): 
        ''' Messy code for dealing with all the different file names 
        '''
        if 'manodeep' not in name: # patchy mocks now are computed using Nbodykit 
            raise NotImplementedError 
        if ireal < 1 or ireal > 50: 
            raise ValueError('only realization between 1-50')
        f = ''.join(['out2_40', str(ireal).zfill(2), '_irot_', str(ibox), 
            '_gmf_geometry_binned.dat']) 
        return f 


class Pk: 
    def __init__(self):
        ''' Pk object
        '''
        self.k = None 
        self.pk = None 
        self.counts = None 

    def Observation(self, ell, zbin, nors):
        ''' Read in P(k) measurements of BOSS from Florian
        '''
        if ell == 0:
            str_pole = 'mono'
        elif ell == 2:
            str_pole = 'quadru'
        elif ell == 4:
            str_pole = 'hexadeca'
        str_pole += 'pole'

        f = ''.join([UT.dat_dir(), 'Beutler/public_material_RSD/',
            'Beutleretal_pk_', str_pole, '_DR12_', nors.upper(), '_z', str(zbin), 
            '_prerecon_120.dat'])
        k_central, k_mean, pk = np.loadtxt(f, skiprows=31, unpack=True, usecols=[0,1,2])
        self.k_obv = k_central 
        self.k_mean_obv = k_mean 
        self.k_central_obv = k_central 
        self.pk_obv = pk
        return k_central, pk 

    def Read(self, name, i, ell=0, sys=None, NorS='ngc'): 
        ''' Read in the i th mock catalog P(k) 
        '''
        if ell not in [0, 2, 4]: 
            raise ValueError("ell can only be 0, 2, or 4") 
        
        #f = self._compressed_read(name, i, ell, sys) 
        f = ''.join([UT.catalog_dir(name), self._file_name(name, i, NorS, sys)]) 

        if 'patchy' in name: 
            k, pk = np.loadtxt(f, skiprows=24, unpack=True, usecols=[0,1+ell/2]) # k, P_l(k) 
        else: 
            if ell == 0: 
                i_ell = 5
            else: 
                i_ell = 1 + ell/2
            k, pk, counts = np.loadtxt(f, unpack=True, usecols=[0, i_ell, -2]) # k, p0(k), and number of modes 

            self.counts = counts
        self.k = k
        self.pk = pk
        return None

    def krange(self, krange): 
        ''' impose some k_max on self.k and self.pk  
        Ideally kmax has to be imposed *before* rebinning.
        '''
        if krange is None: # do nothing
            return None
        kmin, kmax = krange
        if self.k is None: 
            raise ValueError("k and Pk have to be read in")
        if self.pk is None: 
            raise ValueError("k and Pk have to be read in")
        kbin = np.where((self.k >= kmin) & (self.k <= kmax)) 
        k = self.k[kbin]
        pk = self.pk[kbin]

        self.k = k
        self.pk = pk
        if self.counts is not None:
            counts = self.counts[kbin]
            self.counts = counts
        return None 
    
    def _file_name(self, name, i, nors, sys): 
        ''' Messy code for dealing with all the different file names 
        '''
        if 'patchy' in name: # patchy mocks now are computed using Nbodykit 
            if sys != 'fc': 
                raise ValueError
            zbin = name.split('.z')[-1]
            f = ''.join(['pk.patchy.', str(i), '.', nors, '.zbin', str(zbin), '.nbodykit.dat'])
            #f = ''.join(['pk.patchy.', str(i), '.nbodykit.zbin', str(zbin), '.dat'])
        else: 
            if sys is None: 
                str_sys = ''
            elif sys == 'fc':  # fiber collisions 
                if 'patchy' in name: 
                    str_sys = '.fc'
                else: 
                    str_sys = '.fibcoll'
            else: 
                raise NotImplementedError

            if name == 'nseries': 
                f = ''.join(['POWER_Q_CutskyN', str(i), '.fidcosmo', str_sys, '.dat.grid480.P020000.box3600'])
            elif name == 'qpm': 
                f = ''.join(['POWER_Q_a0.6452_', str(i).zfill(4), '.dr12d_cmass_ngc.vetoed.fidcosmo', str_sys, '.dat.grid480.P020000.box3600']) 
            elif 'patchy.ngc.' in name: 
                zbin = name.split('.z')[-1]
                f = ''.join(['plk.Patchy-Mocks-DR12NGC-COMPSAM_V6C_', str(i).zfill(4), 
                    '.Lbox3600.Ngrid480.Nbin40.O4intp.P010000', str_sys, '.z', zbin]) 
                    #'.Lbox2800.Ngrid360.Nbin40.O4intp.P010000', str_sys, '.z', zbin]) 
            else: 
                raise NotImplementedError
        return f 
    
    def _n_mock(self, name): 
        ''' Given name of mock return n_mock. 
        '''
        if name == 'nseries': 
            n_mock = 84
        elif name == 'qpm': 
            n_mock = 1000 
        elif 'patchy' in name: 
            n_mock = 2048
        else: 
            raise ValueError
        return n_mock 
    
    def _tarfile(self, name): 
        ''' tar ball that contains all the powerspectrum measurements. 
        with/without systematics, everything. P(k) files are pretty small 
        so who cares. 
        '''
        return ''.join([UT.catalog_dir(name), 'power_', name, '.tar'])

    def _compressed_read(name, i, ell, sys): 
        ''' Reading from compressed file. I experimented it because
        it seemed cool, but takes a super long time...
        '''
        # compressed file 
        tar = tarfile.open(self._tarfile(name))

        fname = self._file_name(name, i, sys) # file name 
        cnt = 0  
        for mem in tar.getmembers(): 
            if mem.name == fname: 
                member = mem 
                cnt += 1 
                continue 
        if cnt == 0: 
            print(fname)
            print('not in ')
            print(self._tarfile(name))
            raise ValueError
        f = tar.extractfile(member)
        return f 
    
    def _rebin(self, rebin, pk_arr=None): 
        ''' *** Not a great way to re-bin***
        Rebin the P(k) using the counts 

        pk_arr = [k, pk, count] 
        '''
        if pk_arr is None and self.k is None: 
            raise ValueError("what am I rebinning") 
        if pk_arr is not None: 
            if self.k is not None: 
                print("Pk object already has k, Pk, and count. They will be ignored.")
            k = pk_arr[0] 
            pk = pk_arr[1] 
            counts = pk_arr[2]
        else: 
            k = self.k 
            pk = self.pk 
            if self.counts is None:
                raise ValueError
            counts = self.counts

        N = len(k) # array length 

        if isinstance(rebin, int): 
            # reducing binning by a factor or rebin 
            tot_counts, k_rebin, pk_rebin = [], [], []
            for istart in range(N)[::rebin]: 
                indices = range(N)[istart:istart+rebin]

                cnts = np.sum(counts[indices]) # number of modes in bin 
                tot_counts.append(cnts)
                k_rebin.append(np.sum(k[indices] * counts[indices])/cnts)
                pk_rebin.append(np.sum(pk[indices] * counts[indices])/cnts)
            # note that if rebin does not divide evenly into N, 
            # the last bin will be uneven

        elif rebin == 'beutler': 
            # rebin to match Florian's binning 
            k_binedge = np.linspace(0.0, 1.0, 101) # dk = 0.01 
            
            tot_counts, k_rebin, pk_rebin = [], [], []
            for i_bin in range(len(k_binedge)-1): 
                inbin = np.where((k >= k_binedge[i_bin]) & (k < k_binedge[i_bin+1]))
                if len(inbin[0]) > 0: 
                    cnts = np.sum(counts[inbin]) # number of modes in bin 
                    tot_counts.append(cnts)
                    k_rebin.append(np.sum(k[inbin] * counts[inbin])/cnts)
                    pk_rebin.append(np.sum(pk[inbin] * counts[inbin])/cnts)
        else: 
            raise NotImplementedError

        return [np.array(k_rebin), np.array(pk_rebin), np.array(tot_counts)]


def patchyCov(zbin, NorS='ngc', ell=0, clobber=False): 
    ''' Construct covariance matrix for patchy mocks measured using Nbodykit 
    '''
    catalog = 'patchy.z'+str(zbin)

    f_cov = ''.join([UT.catalog_dir(catalog), 'Cov_pk.', catalog, '.ell', str(ell), '.', NorS, '.NBKT.dat']) 
    
    if os.path.isfile(f_cov) and not clobber:  
        i_k, j_k, k_i, k_j, C_ij = np.loadtxt(f_cov, skiprows=4, unpack=True, usecols=[0,1,2,3,-1])

        ki = k_i.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
        kj = k_j.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
        Cij = C_ij.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
        return ki, kj, Cij
    else: 
        # calculate my patchy covariance 
        pkay = Pk() 
        n_mock = pkay._n_mock(catalog) 
        i_mock, n_missing = 0, 0 
        for i in range(1,n_mock+1):
            try: 
                pkay.Read(catalog, i, ell=ell, NorS=NorS, sys='fc')
                pkay.krange([0.01,0.15])
                k = pkay.k
                pk = pkay.pk
                n_kbin = len(k) 
                
                if i == 1: 
                    ks = np.zeros((n_mock, n_kbin))
                    pks = np.zeros((n_mock, n_kbin))
                ks[i_mock,:] = k
                pks[i_mock,:] = pk
                i_mock += 1
            except IOError: 
                if i == 1: 
                    raise ValueError
                print('missing -- ', pkay._file_name(catalog, i, 'fc'))
                n_missing += 1 
        print(n_missing, "P(k) files missing") 
        n_mock -= n_missing
        if n_missing > 0: 
            pks = pks[:n_mock,:]

        #mu_pk = np.sum(pks, axis=0)/np.float(n_mock)
        #Cov_pk = np.dot((pks-mu_pk).T, pks-mu_pk)/float(n_mock-1)
        Cov_pk = np.cov(pks.T)

        # write to file 
        f = open(f_cov, 'w')
        f.write('### header ### \n') 
        f.write('Covariance matrix for ell = '+str(ell)+' calculated from the '+str(n_mock)+' mocks \n')
        f.write('5 columns: {measured power spectrum bin index i} {measured power spectrum bin index j} k_i k_j C_ij \n') 
        f.write('### header ### \n') 
    
        k_i, k_j = [], [] 
        for i in range(Cov_pk.shape[0]): 
            for j in range(Cov_pk.shape[1]):
                f.write('%i \t %i \t %f \t %f \t %e' % (i+1, j+1, k[i], k[j], Cov_pk[i,j])) 
                f.write('\n') 
                k_i.append(k[i])
                k_j.append(k[j])
        f.close() 
        k_i = np.array(k_i)
        k_j = np.array(k_j)
        ki = k_i.reshape(Cov_pk.shape)
        kj = k_j.reshape(Cov_pk.shape)
        return ki, kj, Cov_pk 


def beutlerCov(zbin, NorS='ngc', ell=0): 
    ''' Read in Florian's covariance matrix
    '''
    # read in C_ij from file 
    if NorS == 'ngc': 
        f_cov = ''.join([UT.dat_dir(), 'Beutler/public_material_RSD/',
            'Beutleretal_cov_patchy_z', str(zbin), '_NGC_1_15_1_15_1_10_2045_60.dat']) 
    else: 
        f_cov = ''.join([UT.dat_dir(), 'Beutler/public_material_RSD/',
            'Beutleretal_cov_patchy_z', str(zbin), '_SGC_1_15_1_15_1_10_2048_60.dat']) 
    i_k, j_k, k_i, k_j, C_ij = np.loadtxt(f_cov, skiprows=4, unpack=True, usecols=[0,1,2,3,4])
    
    if ell == 0: 
        i_l0, i_l1 = 0, 14
    elif ell == 2: 
        i_l0, i_l1 = 14, 28 
    elif ell == 4: 
        i_l0, i_l1 = 28, 37 
    elif ell == 'all': 
        i_l0, i_l1 = 0, 37 

    ki = k_i.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    kj = k_j.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    Cij = C_ij.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    #return C_ij.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    return ki[i_l0:i_l1,i_l0:i_l1], kj[i_l0:i_l1,i_l0:i_l1], Cij[i_l0:i_l1,i_l0:i_l1]
