
import numpy as np

import algos


class PLA(algos.Iterative):
    '''
    Perceptron learning algorithm.
    '''
    
    ## Clerical methods. ##

    def __init__(self, w_init, t_max=None, thres=None,
                 verbose=False, store=False, name="PLA"):

        super(PLA, self).__init__(w_init=w_init, t_max=t_max,
                                  thres=thres, verbose=verbose,
                                  store=store, name=name)


    ## Core update implementation. ##
    
    def update(self, model, data):

        if model is None or data is None:
            raise ValueError("Need model and data to update.")
        
        n, d = data.X_tr.shape

        loss_01 = model.l_tr(w=self.w, data=data).flatten()
        fail_count = np.sum(loss_01)
        
        if fail_count == 0:
            self.early_stop = True
            if self.verbose:
                print("Error free. Stopping early (t = {}).".format(self.t))
        else:

            if self.verbose:
                print("Number of error: {} of {}.".format(fail_count,n))

            # Choose one mis-classified point randomly.
            idx_fail = np.arange(n)[loss_01 > 0]
            idx = np.random.choice(a=idx_fail, size=1)
            x_fail = np.take(a=data.X_tr, indices=idx, axis=0)
            y_fail = data.y_tr[idx,0]
            
            if y_fail == 1:
                sign_fail = 1
            else:
                sign_fail = -1
            
            # Then update using this point.
            self.w += sign_fail * x_fail.T

        # Update various elements being monitored.
        self._monitor(model=model, data=data)
        

    
