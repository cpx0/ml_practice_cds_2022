
import numpy as np

import algos


class TightestFit(algos.Direct):
    '''
    Tightest-fit rectangle on the 2D plane,
    assuming sides are parallel to horizontal
    and vertical axes.
    '''
    
    ## Clerical methods. ##

    def __init__(self, name="TightestFit"):

        super(TightestFit, self).__init__(name=name)


    ## Core update implementation. ##
    
    def run(self, model, data):
        
        if model is None or data is None:
            raise ValueError("Need model and data to update.")
        
        n, d = data.X_tr.shape
        
        idx_in = data.y_tr[:,0] == 1
        
        mins = [np.min(data.X_tr[idx_in,0]),np.min(data.X_tr[idx_in,1])]
        maxes = [np.max(data.X_tr[idx_in,0]),np.max(data.X_tr[idx_in,1])]
        
        maxmin = {"mins":mins, "maxes":maxes}
        
        return maxmin

        

    
