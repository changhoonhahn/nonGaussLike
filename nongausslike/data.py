'''

code for assessing galaxy clustering data and etc. 


'''
import os 
import numpy as np 
import tarfile 
# -- local -- 
import util as UT

from ChangTools.fitstables import mrdfits


class Pk: 
    def __init__(self):
        ''' Pk object
        '''
        self.k = None 
        self.pk = None 
        self.counts = None 

    def Read(self, name, i, ell=0, sys=None): 
        ''' Read in the i th mock catalog P(k) 
        '''
        if ell not in [0, 2, 4]: 
            raise ValueError("ell can only be 0, 2, or 4") 
        
        #f = self._compressed_read(name, i, ell, sys) 
        f = ''.join([UT.catalog_dir(name), self._file_name(name, i, sys)]) 
    
        if ell == 0: 
            i_ell = 5
        else: 
            i_ell = 1 + ell/2
    
        k, pk, counts = np.loadtxt(f, unpack=True, usecols=[0, i_ell, -2]) # k, p0(k), and number of modes 

        self.k = k
        self.pk = pk
        self.counts = counts
        return None

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
            print fname 
            print 'not in ' 
            print self._tarfile(name)
            raise ValueError
        f = tar.extractfile(member)
        return f 
    
    def rebin(self, rebin, pk_arr=None): 
        ''' Rebin the P(k) using the counts 

        pk_arr = [k, pk, count] 
        '''
        if pk_arr is None and self.k is None: 
            raise ValueError("what am I rebinning") 
        if pk_arr is not None: 
            if self.k is not None: 
                print "Pk object already has k, Pk, and count. They will be ignored."
            k = pk_arr[0] 
            pk = pk_arr[1] 
            counts = pk_arr[2]
        else: 
            k = self.k 
            pk = self.pk 
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
        counts = self.counts[kbin]

        self.k = k
        self.pk = pk
        self.counts = counts
        return None 
    
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

    def _file_name(self, name, i, sys): 
        ''' Messy code for dealing with all the different file names 
        '''
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
        else: 
            raise NotImplementedError
        return f 


def patchyCov(zbin, NorS='ngc', ell=0, clobber=False): 
    ''' Construct covariance matrix for patchy mocks measured using Roman's code
    and compare with Florian's. 
    '''
    catalog = 'patchy.'+NorS+'.z'+str(zbin)

    f_cov = ''.join([UT.catalog_dir(catalog), 'Cov_pk.', catalog, '.ell', str(ell), '.beutler.dat']) 
    
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
        n_missing = 0 
        for i in range(1,n_mock+1):
            try: 
                # read in monopole
                pkay.Read(catalog, i, ell=ell, sys='fc')
                pkay.krange([0.01,0.15])
                k = pkay.k
                pk = pkay.pk
                #k0, p0k, _ = pkay.rebin('beutler') 
                n_kbin = len(k) 
                
                if i == 1: 
                    ks = np.zeros((n_mock, n_kbin))
                    pks = np.zeros((n_mock, n_kbin))

                ks_i, pks_i = k, pk 
                ks[i-1,:] = ks_i
                pks[i-1,:] = pks_i 
            except IOError: 
                if i == 1: 
                    raise ValueError
                print 'missing -- ', pkay._file_name(catalog, i, 'fc')
                n_missing += 1 

        n_mock -= n_missing
        if n_missing > 0: # just a way to deal with missing  
            pks = pks[:n_mock,:]

        Cov_pk = np.cov(pks.T)
        #f_hartlap = float(pks.shape[0] - pks.shape[1] - 2) / float(pks.shape[0] - 1) 
        #Cov_pk *= 1./f_hartlap 

        # write to file 
        f = open(f_cov, 'w')
        f.write('### header ### \n') 
        f.write('Covariance matrix for ell = '+str(ell)+' calculated from the '+str(n_mock)+' mocks \n')
        f.write('5 columns: {measured power spectrum bin index i} {measured power spectrum bin index j} k_i k_j C_ij \n') 
        f.write('### header ### \n') 
    
        k_i, k_j = [], [] 
        for i in range(Cov_pk.shape[0]): 
            for j in range(Cov_pk.shape[1]):
                f.write('%i \t %i \t %f \t %f \t %e' % (i+1, j+1, ks_i[i], ks_i[j], Cov_pk[i,j])) 
                f.write('\n') 
                k_i.append(ks_i[i])
                k_j.append(ks_i[j])
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

    ki = k_i.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    kj = k_j.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    Cij = C_ij.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    #return C_ij.reshape((int(np.sqrt(len(i_k))), int(np.sqrt(len(i_k)))))
    return ki[i_l0:i_l1,i_l0:i_l1], kj[i_l0:i_l1,i_l0:i_l1], Cij[i_l0:i_l1,i_l0:i_l1]
