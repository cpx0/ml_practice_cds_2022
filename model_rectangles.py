
import numpy as np

import models


class Rectangle(models.Classifier):
    '''
    Rectangles on the plane.
    Original example via Blumer et al. (1989).
    '''
    
    ## Clerical methods. ##

    def __init__(self, data=None, name="Rectangle"):
        
        super(Rectangle, self).__init__(data=data, name=name)


    def __call__(self, X, minmax, pmone=False):

        if pmone:
            return self._classify_pmone(X=X, minmax=minmax)
        else:
            return self._classify(X=X, minmax=minmax)


    ## Methods related to execution. ##
    
    def _classify(self, X, minmax):
        '''
        X is a (k x 2) matrix of k observations on 2D plane.
        Returns array of shape (k x 1) of predicted labels.
        '''
        
        mincheck = np.sum((X < minmax["mins"]), axis=1)
        maxcheck = np.sum((minmax["maxes"] < X), axis=1)
        
        return np.where((mincheck+maxcheck) > 0, 0, 1).reshape((X.shape[0],1))


    def _classify_pmone(self, X, w):
        '''
        X is a (k x 2) matrix of k observations on 2D plane.
        Returns array of shape (k x 1) of predicted labels.
        '''
        
        mincheck = np.sum((X < minmax["mins"]), axis=1)
        maxcheck = np.sum((minmax["maxes"] < X), axis=1)
        
        return np.where((mincheck+maxcheck) > 0, -1, 1).reshape((X.shape[0],1))


    ## Methods related to feedback. ##

    def l_imp(self, minmax=None, X=None, y=None, paras=None):
        '''
        Returns the zero-one loss.
        '''
        
        yhat = self._classify(X=X, minmax=minmax)
        return np.where(yhat != y, 1, 0)



        
