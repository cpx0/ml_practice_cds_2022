
import numpy as np

import data
import model_perceptron
import helpers as hlp


# Parameters for experimental setup.
_x_loc = 0.0
_x_scale = 1.0
_n_tr = 20
_n_te = 80
_w_star = np.array([-0.25, -0.5, 0.25]).reshape((3,1))
_d = _w_star.size - 1


def gen():
    '''
    Data generation function, simplest example.
    All such functions are defined in such a way
    that they return a DataSet object.
    '''

    # Generate the inputs randomly.
    
    X_tr = np.random.normal(loc=_x_loc, scale=_x_scale,
                            size=(_n_tr, _d))
    X_tr = hlp._concat_ones(X=X_tr)

    X_te = np.random.normal(loc=_x_loc, scale=_x_scale,
                            size=(_n_te, _d))
    X_te = hlp._concat_ones(X=X_te)

    # Classify according to a pre-set hyperplane.
    
    model = model_perceptron.Perceptron()
    
    y_tr = model(X=X_tr, w=_w_star)
    y_te = model(X=X_te, w=_w_star)

    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="perceptron_noisefree")


