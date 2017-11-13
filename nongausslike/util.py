'''

General utility functions 

'''
import os
import sys
import numpy as np
from scipy import interpolate


def code_dir(): 
    ''' Directory where all the code is located (the directory that this file is in!)
    '''
    return os.path.dirname(os.path.abspath(__file__))+'/'


def catalog_dir(name): 
    ''' sub directory of dat where the specified catalog is located
    '''
    name_dir = name 
    if 'patchy' in name: 
        name_dir = 'patchy'
    elif 'manodeep' in name: 
        name_dir = ''.join([name.split('.run')[0], '/run_', name.split('.run')[-1].zfill(4)])
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/dat/'+name_dir+'/'


def dat_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/dat/'


def fig_dir(): 
    ''' 
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/figs/'


def tex_dir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/paper/'


def run_dir(): 
    ''' 
    '''
    return code_dir()+'run/'


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
