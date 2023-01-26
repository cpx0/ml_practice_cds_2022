
import numpy as np

import data
import model_perceptron
import helpers as hlp


# Parameters for experimental setup.

_x_loc = 0.0
_x_scale = 1.0
_noise_loc = 0.0
_noise_scale = 0.1

def gen(n_tr, n_te, d, noise_free=False):
    '''
    Essentially the same as data_PLA_noisy, except
    generalized to arbitrary dimension, controllable
    at runtime, with the option to have no noise.
    The w_star hyperplane is now generated randomly.
    '''

    # Generate the inputs randomly.
    
    X_tr = np.random.normal(loc=_x_loc, scale=_x_scale,
                            size=(n_tr, d))
    X_tr = hlp._concat_ones(X=X_tr)

    X_te = np.random.normal(loc=_x_loc, scale=_x_scale,
                            size=(n_te, d))
    X_te = hlp._concat_ones(X=X_te)

    # Generate the classifying hyperplane randomly.

    w_star = np.random.uniform(size=(d+1,1))

    # Classify assuming an additive noise model.

    model = model_perceptron.Perceptron()
    
    if noise_free:
        noise_tr = 0.0
        noise_te = 0.0
    else:
        noise_tr = np.random.normal(loc=_noise_loc, scale=_noise_scale,
                                    size=(n_tr, 1))
        noise_te = np.random.normal(loc=_noise_loc, scale=_noise_scale,
                                    size=(n_te, 1))
    
    score_tr = model._score(X=X_tr, w=w_star) + noise_tr
    score_te = model._score(X=X_te, w=w_star) + noise_te
    
    y_tr = np.where(score_tr <= 0, 0, 1)
    y_te = np.where(score_te <= 0, 0, 1)
    
    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="perceptron_generic")


