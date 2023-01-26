
import numpy as np


class DataSet:
    '''
    Base class for data objects.
    '''
    
    ## Clerical methods. ##
    
    def __init__(self, X_tr=None, X_te=None, y_tr=None, y_te=None, name=""):

        self.init_tr(X=X_tr, y=y_tr)
        self.init_te(X=X_te, y=y_te)
        self.name = name
        

    def __str__(self):
        
        out = "Dataset name: {}".format(self.name)
        return out


    ## Methods for populating data attributes. ##
    
    def init_tr(self, X, y=None):
        
        self.X_tr = X
        self.y_tr = y
        if X is not None:
            self.n_tr, self.d_tr = X.shape


    def init_te(self, X, y=None):
        
        self.X_te = X
        self.y_te = y
        if X is not None:
            self.n_te, self.d_te = X.shape

