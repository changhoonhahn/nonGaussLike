import numpy as np
from scipy.linalg import sqrtm
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def KL_w2009_eq5(X, Y, k=5): 
    ''' kNN KL divergence estimate using Eq. 5 from Wang et al. (2009)  

    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    d, n, m = XY_dim(X, Y)

    NN_X = NearestNeighbors(n_neighbors=k+1).fit(X)
    NN_Y = NearestNeighbors(n_neighbors=k).fit(Y)
    dNN_XX, _ = NN_X.kneighbors(X, n_neighbors=k+1)
    dNN_XY, _ = NN_Y.kneighbors(X, n_neighbors=k)
    rho_i = dNN_XX[:,-1]
    nu_i = dNN_XY[:,-1]
    return float(d)/float(n) * np.sum(np.log(nu_i / rho_i)) + np.log(float(m)/float(n - 1))


def KL_w2009_eq25(X, Y): 
    ''' kNN KL divergence estimate using Eq. 25 from Wang et al. (2009). 
    This has some bias reduction applied to it. 

    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    d, n, m = XY_dim(X, Y)
    
    # first determine epsilon(i) 
    NN_X = NearestNeighbors(n_neighbors=1).fit(X)
    NN_Y = NearestNeighbors(n_neighbors=1).fit(Y)
    dNN1_XX, _ = NN_X.kneighbors(X, n_neighbors=2)
    dNN1_XY, _ = NN_Y.kneighbors(X)   
    eps = np.amax([dNN1_XX[:,1], dNN1_XY[:,0]], axis=0) * 1.000001
    
    # find l_i and k_i
    _, i_l = NN_X.radius_neighbors(X, eps)
    _, i_k = NN_Y.radius_neighbors(X, eps)
    l_i = np.array([len(il)-1 for il in i_l])
    k_i = np.array([len(ik) for ik in i_k])
    return np.sum(digamma(l_i) - digamma(k_i)) / float(n) + np.log(float(m)/float(n-1))


def KL_w2009_eq29(X, Y):
    ''' kNN KL divergence estimate using Eq. 29 from Wang et al. (2009). 
    This has some bias reduction applied to it and a correction for 
    epsilon.

    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    d, n, m = XY_dim(X, Y)

    # first determine epsilon(i)
    NN_X = NearestNeighbors(n_neighbors=1).fit(X)
    NN_Y = NearestNeighbors(n_neighbors=1).fit(Y)
    dNN1_XX, _ = NN_X.kneighbors(X, n_neighbors=2)
    dNN1_XY, _ = NN_Y.kneighbors(X)
    eps = np.amax([dNN1_XX[:,1], dNN1_XY[:,0]], axis=0) * 1.000001

    # find l_i and k_i
    _, i_l = NN_X.radius_neighbors(X, eps)
    _, i_k = NN_Y.radius_neighbors(X, eps)
    l_i = np.array([len(il)-1 for il in i_l])
    k_i = np.array([len(ik) for ik in i_k])
    #assert l_i.min() > 0
    #assert k_i.min() > 0

    rho_i = np.empty(n, dtype=float)
    nu_i = np.empty(n, dtype=float)
    for i in range(n):
        rho_ii, _ = NN_X.kneighbors(np.atleast_2d(X[i]), n_neighbors=l_i[i]+1)
        nu_ii, _ = NN_Y.kneighbors(np.atleast_2d(X[i]), n_neighbors=k_i[i])
        rho_i[i] = rho_ii[0][-1]
        nu_i[i] = nu_ii[0][-1]

    d_corr = float(d) / float(n) * np.sum(np.log(nu_i/rho_i))
    return d_corr + np.sum(digamma(l_i) - digamma(k_i)) / float(n) + np.log(float(m)/float(n-1))


def KL_w2009_eq5W(X, Y, k=5): 
    ''' kNN KL divergence estimate using Eqs. 30-33 from Wang et al. (2009). 
    Whiten X and Y first before using the kNN estimator on it  

    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    d, n, m = XY_dim(X, Y)

    mu = 1./float(n + m) * (np.sum(X, axis=0) + np.sum(Y, axis=0))

    _Cx = np.sum([np.dot((X[i] - mu)[:,None], (X[i] - mu)[None,:]) for i in range(n)], axis=0)
    _Cy = np.sum([np.dot((Y[i] - mu)[:,None], (Y[i] - mu)[None,:]) for i in range(m)], axis=0)
    C = 1./float(n + m - 1) * (_Cx + _Cy)

    Cinvsqrt = np.linalg.inv(sqrtm(C))
    Xp = np.dot(Cinvsqrt, (X - mu).T).T
    Yp = np.dot(Cinvsqrt, (Y - mu).T).T
    return KL_w2009_eq5(Xp, Yp, k=k)


def XY_dim(X, Y): 
    assert X.shape[1] == Y.shape[1]
    d = X.shape[1] # dimensions
    n = X.shape[0] # X sample size
    m = Y.shape[0] # Y sample size
    return d, n, m 
