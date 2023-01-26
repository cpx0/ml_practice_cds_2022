
import numpy as np

import models


class Perceptron(models.Classifier):
    '''
    Classical perceptron model.
    '''

    ## Clerical methods. ##

    def __init__(self, data=None, name="Perceptron"):

        super(Perceptron, self).__init__(data=data, name=name)


    def __call__(self, X, w, pmone=False):

        if pmone:
            return self._classify_pmone(X=X, w=w)
        else:
            return self._classify(X=X, w=w)


    ## Methods related to execution. ##
    
    def _score(self, X, w):
        '''
        Assign real-valued score to inputs.
        w is a (d x 1) array of weights.
        X is a (k x d) matrix of k observations.
        Returns array of shape (k x 1) of predicted values.
        '''
        
        return np.matmul(X,w)
    
    
    def _classify(self, X, w):
        '''
        Binary classification based on the sign of the score,
        following the encoding requiring labels to be in {0,1}.
        Has same shape as the output of _score.
        '''

        return np.where(self._score(X=X, w=w) <= 0, 0, 1)


    def _classify_pmone(self, X, w):
        '''
        Analogue to _classify, but with outputs in {-1,1}.
        '''

        return np.where(self._score(X=X, w=w) <= 0, -1, 1)


    ## Methods related to feedback. ##

    def l_imp(self, w=None, X=None, y=None, paras=None):
        '''
        Returns the zero-one loss.
        '''
        
        yhat = self._classify(X=X, w=w)
        return np.where(yhat != y, 1, 0)



        
