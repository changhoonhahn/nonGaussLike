'''

plots for the paper  


'''
import numpy as np 
from scipy.stats import gaussian_kde as gkde
from scipy.stats import multivariate_normal as mGauss
# -- local -- 
import data as Data
import util as UT 
import nongauss as NG 
# -- plotting -- 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors
