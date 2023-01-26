
import numpy as np

import algos


class OLS(algos.Direct):
    '''
    Ordinary least squares.
    '''
    
    ## Clerical methods. ##

    def __init__(self, name="OLS"):
        
        super(OLS, self).__init__(name=name)


    ## Core implementation. ##
    
    def run(self, model=None, data=None):
        
        self.w = np.linalg.lstsq(a=data.X_tr, b=data.y_tr, rcond=None)[0]
        

class Ridge(algos.Direct):
    '''
    Classical ridge regression in one step.
    '''
    
    ## Clerical methods. ##

    def __init__(self, lamreg, name="OLS"):
        
        super(Ridge, self).__init__(name=name)
        
        self.lamreg = lamreg


    ## Core implementation. ##
    
    def run(self, model=None, data=None):
        n, d = data.X_tr.shape
        inv = np.linalg.inv(a=(np.matmul(data.X_tr.T, data.X_tr)+self.lamreg*np.eye(N=d)))
        self.w = np.matmul(np.matmul(inv, data.X_tr.T), data.y_tr)
        
        
        
        
        
