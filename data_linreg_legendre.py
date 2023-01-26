
import numpy as np

import data
import helpers as hlp


# Parameters for experimental setup.

def _f_true(x):
    '''
    A simple test function, non-linear,
    to be learned using a linear model with
    Legendre polynomial features.
    '''
    return (np.abs(x-0.3))**2/2 + 0.3

_x = np.array([-2., -1.2, -1.0, 0.5, 2.1])
_n = len(_x)


def gen():
    '''
    Data generation function, simplest example.
    All such functions are defined in such a way
    that they return a DataSet object.
    '''
    
    X_tr = _x.reshape((_n,1))
    
    y_tr = _f_true(X_tr) + np.random.normal(size=X_tr.shape)/2
    
    return data.DataSet(X_tr=X_tr, y_tr=y_tr,
                        name="linreg_legendre")


