
import numpy as np

import data
import model_perceptron
import helpers as hlp


# Parameters for experimental setup.
_x_loc = 0.0
_x_scale = 1.0
_noise_loc = 0.0
_noise_scale = 0.25
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
    
    # Classify assuming an additive noise model.
    
    model = model_perceptron.Perceptron()
    noise_tr = np.random.normal(loc=_noise_loc, scale=_noise_scale,
                                size=(_n_tr, 1))
    noise_te = np.random.normal(loc=_noise_loc, scale=_noise_scale,
                                size=(_n_te, 1))
    
    score_tr = model._score(X=X_tr, w=_w_star) + noise_tr
    score_te = model._score(X=X_te, w=_w_star) + noise_te

    y_tr = np.where(score_tr <= 0, 0, 1)
    y_te = np.where(score_te <= 0, 0, 1)

    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="perceptron_noisy")


