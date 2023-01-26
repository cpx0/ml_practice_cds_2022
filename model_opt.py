
import numpy as np

import models


class Quadratic1D(models.Function):
    '''
    1-dim quadratic function.
    '''
    
    ## Clerical methods. ##

    def __init__(self, shift=0.0, scale=0.5, name="Quadratic1D"):
        
        super(Quadratic1D, self).__init__(name=name)
        
        # Function is fully determined by user-passed parameters.
        self.shift = shift
        self.scale = scale
    
    
    def f_opt(self, w):
        
        # Keep shape of w here for vectorizing.
        return (w-self.shift)**2 * self.scale
    
    
    def g_opt(self, w):
        
        # Keep shape of w here.
        return 2*(w-self.shift)*self.scale
    
    
class Poly15(models.Function):
    '''
    Polynomial between absolute value and quadratic.
    '''
    
    ## Clerical methods. ##

    def __init__(self, shift=0.0, scale=2./3., name="Poly15"):
        
        super(Poly15, self).__init__(name=name)
        
        # Function is fully determined by user-passed parameters.
        self.shift = shift
        self.scale = scale
    
    
    def f_opt(self, w):
        
        # Keep shape of w here for vectorizing.
        return np.abs(w-self.shift)**(1.5) * self.scale
    
    
    def g_opt(self, w):
        
        # Keep shape of w here.
        topass = w-self.shift
        return 1.5*np.sign(topass)*np.sqrt(np.abs(topass))*self.scale




        
