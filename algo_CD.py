
import numpy as np

import algos
import helpers as hlp



class CD_LASSO_ERM(algos.Iterative):
    '''
    Coordinate descent (CD) for the LASSO, using ERM,
    i.e., average over the data points in the sample.
    '''
    
    ## Clerical methods. ##

    def __init__(self, w_init, t_max=None, thres=None,
                 verbose=False, store=False, paras=None, name="CD_LASSO_ERM"):

        super(CD_LASSO_ERM, self).__init__(w_init=w_init,
                                           t_max=t_max, thres=thres,
                                           verbose=verbose,
                                           store=store, name=name)
        
        self.paras = paras
        self.idx = np.random.choice(w_init.size, size=w_init.size, replace=False)
        self.idx_j = self.idx[0]
    
    
    ## Implementation of new update. ##
    
    def update(self, model=None, data=None):

        if model is None:
            raise ValueError("At least need model to update.")

        # Parameter update.
        n = data.X_tr.shape[0]
        idx_mod = self.t % self.w.size
        self.idx_j = self.idx[idx_mod] # circuits around shuffled coords.
        self.w[self.idx_j,0] = 0 # current para, but with jth coord set to zero.
        g_j = -np.mean(model.g_j_tr(j=self.idx_j, w=self.w, data=data, paras=None))
        g_j = g_j * n / (n-1) # re-scale.
        
        # Compute the solution to the one-dimensional optimization, using it to
        # update the parameter at the specified coordinate.
        self.w[self.idx_j,0] = hlp.soft_thres(u=g_j, mar=self.paras["lamreg"])
        
        # Update various elements being monitored.
        self._monitor(model=model, data=data)





