
import numpy as np

import models
import model_perceptron


class Hinge(model_perceptron.Perceptron):
    '''
    Perceptron model with hinge-based feedback.
    '''
    
    ## Clerical methods. ##

    def __init__(self, data=None, name="Perceptron-Hinge"):
        
        super(Hinge, self).__init__(data=data, name=name)


    ## Methods related to feedback. ##

    def l_imp(self, w=None, X=None, y=None, paras=None):
        '''
        Returns the hinge loss.
        '''
        
        y_pmone = np.where(y==0, -1, y) # assumes labels in {0,1}.
        
        if paras is not None:
            penalty = np.sum(w**2) * paras["reg_coef"] / 2
        else:
            penalty = 0.0
        
        margins_gap = 1 - y_pmone * self._score(X=X, w=w)
        return np.where(margins_gap > 0, margins_gap, 0) + penalty


    def g_imp(self, w=None, X=None, y=None, paras=None):
        '''
        Gradient of the hinge loss.
        '''
        
        y_pmone = np.where(y==0, -1, y) # assumes labels in {0,1}.
        
        if paras is not None:
            g_penalty = paras["reg_coef"] * w.T
        else:
            g_penalty = 0.0
        
        out = -y_pmone*X
        margins_gap = 1 - y_pmone * self._score(X=X, w=w)
        idx_flat = margins_gap.flatten() <= 0
        out[idx_flat,:] = 0
        
        return out + g_penalty
        



    
