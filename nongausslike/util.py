'''

General utility functions 

'''
import os
import sys
import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde as gKDE


def check_env(): 
    if os.environ.get('NONGAUSSLIKE_DIR') is None: 
        raise ValueError("set $NONGAUSSLIKE_DIR in bashrc file!") 
    return None


class KayDE(gKDE): 
    def __init__(self, X): 
        ''' wrapper for gKDE to behave more like 
        sklearn.mixture.GaussianMixutre
        '''
        super(KayDE, self).__init__(X.T)

    def sample(self, N): 
        samp = super(KayDE, self).resample(N) 
        return samp.T

    def score_sample(self, x):  
        if len(x.shape) == 2: 
            if x.shape[1] == 1: 
                x = x[:,0]
        return super(KayDE, self).logpdf(x) 


def code_dir(): 
    ''' Directory where all the code is located (the directory that this file is in!)
    '''
    return os.environ.get('NONGAUSSLIKE_CODEDIR') 


def catalog_dir(name): 
    ''' sub directory of dat where the specified catalog is located
    '''
    name_dir = name 
    if 'patchy' in name: 
        name_dir = 'patchy'
    elif 'manodeep' in name: 
        name_dir = ''.join([name.split('.run')[0], '/run_', name.split('.run')[-1].zfill(4)])
    #return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/dat/'+name_dir+'/'
    return os.environ.get('NONGAUSSLIKE_DIR')+name_dir+'/'


def dat_dir(): 
    ''' 
    '''
    return os.environ.get('NONGAUSSLIKE_DIR') 


def fig_dir(): 
    return os.environ.get('NONGAUSSLIKE_CODEDIR')+'figs/'


def tex_dir():
    return os.environ.get('NONGAUSSLIKE_CODEDIR')+'paper/'


def bar_plot(values, bin_edges): 
    ''' Take outputs from numpy histogram and return pretty bar plot
    '''
    xx = [] 
    yy = [] 

    for i_val, val in enumerate(values): 
        xx.append(bin_edges[i_val]) 
        yy.append(val)
        xx.append(bin_edges[i_val+1]) 
        yy.append(val)

    return [np.array(xx), np.array(yy)]


def gauss(x, sigma, mu): 
    return 1./(np.sqrt(2.*np.pi)*sigma) * np.exp((-(x - mu)**2)/(2*sigma**2))
