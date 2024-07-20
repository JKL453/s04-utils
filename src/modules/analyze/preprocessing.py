"""
preprocess.py - Module for preprocessing timetrace data

Insert description here.

Functions:
- ...

Classes:
- ...

"""

# import statements
from scipy import sparse, signal
from scipy.sparse import linalg
import numpy as np
from numpy.linalg import norm


def baseline_arPLS(signal, ratio=1e-6, lam=100, niter=10, full_output=False):
    '''
    Baseline correction using asymmetrically reweighted penalized least squares smoothing.
    This function is based on the paper by Dieterle et al. (2006). The paper can be found
    at https://pubs.acs.org/doi/pdf/10.1021/ac051632c.
    '''
    
    L = len(signal)
    
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * signal)
        d = signal - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z


