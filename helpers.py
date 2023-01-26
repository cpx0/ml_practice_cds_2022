
import numpy as np


def _concat_ones(X):
    '''
    Concatenates a column vector of ones, in leftmost slot.
    '''
    
    ones = np.ones((X.shape[0],1), dtype=X.dtype)
    return np.concatenate((ones,X), axis=1)


def soft_thres(u, mar):
    '''
    The so-called "soft threshold" function, which
    appears when evaluating the proximal operator
    for a smooth function plus l1-norm.

    Input "u" will be an array, and "mar" will be the
    margin of the soft-threshold, a non-negative real
    value.
    '''
    return np.sign(u) * np.clip(a=(np.abs(u)-mar), a_min=0, a_max=None)


def corr(w, X, y):
    '''
    Wrapper for Pearson's correlation coefficient,
    computed for the predicted response and the
    actual response.

    Args:
    w is a (d x 1) matrix taking real values.
    X is a (k x d) matrix of n observations.
    y is a (k x 1) matrix taking real values.
    
    Output: a real-valued correlation coefficient.
    '''
    yest = np.dot(X,w)
    if np.sum(np.abs(yest)) < 0.0001:
        return 0.0
    else:
        return stats.pearsonr(yest.flatten(), y.flatten())[0]
