
import numpy as np

import data
import model_rectangles
import helpers as hlp


# Parameters for experimental setup.
_x_low = 0.0
_x_high = 1.0
_y_low = 0.0
_y_high = 1.0
_n_tr = 100
_n_te = 1000
_minmax_star = {"mins": (0.05, 0.75),
                "maxes": (0.35, 0.90)}


def gen():
    '''
    Data generation function, simplest example.
    All such functions are defined in such a way
    that they return a DataSet object.
    '''

    # Generate the inputs randomly.
    
    horiz_tr = np.random.uniform(low=_x_low, high=_x_high,
                                 size=(_n_tr, 1))
    vert_tr = np.random.uniform(low=_y_low, high=_y_high,
                                 size=(_n_tr, 1))
    X_tr = np.concatenate((horiz_tr,vert_tr), axis=1)
    
    horiz_te = np.random.uniform(low=_x_low, high=_x_high,
                                 size=(_n_te, 1))
    vert_te = np.random.uniform(low=_y_low, high=_y_high,
                                 size=(_n_te, 1))
    X_te = np.concatenate((horiz_te,vert_te), axis=1)
    
    model = model_rectangles.Rectangle()
    
    y_tr = model(X=X_tr, minmax=_minmax_star)
    y_te = model(X=X_te, minmax=_minmax_star)
    
    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="rectangles_noisefree")


