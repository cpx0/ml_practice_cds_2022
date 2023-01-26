
import numpy as np


class Direct:
    '''
    Base class for algorithms which based on data or some
    initialization parameters directly compute a final
    solution, without need for iteration.
    '''

    ## Clerical methods. ##

    def __init__(self, name=""):

        # Check the user-passed attributes.
        
        self.name = name
        
        # Attributes determined internally.
        ## None at present.
            
    def __str__(self):
        
        out = "Algorithm name: {}".format(self.name)
        return out
    

    def run(self, model=None, data=None):
        
        raise NotImplementedError("To be implemented by sub-classes.")


class Iterative:
    '''
    Simple algorithm base class, for iterative procedures.
    '''

    ## Clerical methods. ##

    def __init__(self, w_init, t_max=None, thres=None,
                 verbose=False, store=False, name=""):

        # Check the user-passed attributes.
        
        self.w = np.copy(w_init)
        self.t_max = t_max
        self.thres = thres
        self.verbose = verbose
        self.store = store
        self.name = name

        if self.t_max is None and self.thres is None:
            raise ValueError("Must provide at least one stopping condition.")
        
        # Attributes determined internally.

        self.early_stop = False
        
        self.diff = np.inf # to record difference with previous step.
        if self.thres is None:
            self.w_old = None
            self.thres = -1.0
        else:
            self.w_old = np.copy(self.w)

        self.t = None # to count the number of steps.
        if self.t_max is None:
            self.t_max = np.inf
            
        if self.store and self.t_max is not None: # for storing states.
            self.wstore = np.zeros((self.w.size,t_max+1), dtype=self.w.dtype)
            self.wstore[:,0] = self.w.flatten()
        else:
            self.wstore = None

            
    def __str__(self):
        
        out = "Algorithm name: {}".format(self.name)
        return out


    def __iter__(self):

        self.t = 0
        
        if self.verbose:
            print("(via __iter__)")
            self._print_state()
            
        return self


    def __next__(self):
        '''
        Check the stopping condition(s).
        '''
        
        if self.t >= self.t_max:
            raise StopIteration
        
        if self.diff < self.thres:
            raise StopIteration

        if self.early_stop:
            raise StopIteration

        if self.verbose:
            print("(via __next__)")
            self._print_state()

    
    def _print_state(self):

        print("------------")
        
        t_state = "t = {} (max = {})".format(self.t, self.t_max)
        print(t_state)
        
        diff_state = "diff = {} (thres = {})".format(self.diff, self.thres)
        print(diff_state)
        
        print("w = ", self.w)
        
        print("------------")
        

    def update(self, model=None, data=None):

        raise NotImplementedError("To be implemented by sub-classes.")
    
    
    def _monitor(self, model=None, data=None):
        
        self.t += 1 # Always record step increase.
        
        if self.thres > 0.0: # Record differences when desired.
            self.diff = np.linalg.norm((self.w-self.w_old))
            self.w_old = np.copy(self.w)
            
        if self.wstore is not None:
            self.wstore[:,self.t] = self.w.flatten()
        

class LineSearch(Iterative):
    '''
    Basic archetype of an iterator for implementing
    line search optimization routines.
    '''

    ## Clerical methods. ##

    def __init__(self, w_init, step, t_max=None, thres=None,
                 verbose=False, store=False, name="LineSearch"):

        super(LineSearch, self).__init__(w_init=w_init, t_max=t_max,
                                         thres=thres, verbose=verbose,
                                         store=store, name=name)

        self.step = step


    def update(self, model=None, data=None):

        if model is None:
            raise ValueError("At least need model to update.")

        # Parameter update.
        newdir = self.newdir(model=model, data=data)
        stepsize = self.step(t=self.t, model=model, data=data, newdir=newdir)
        self.w += stepsize * newdir.T

        # Update various elements being monitored.
        self._monitor(model=model, data=data)
        

    def newdir(self, model, data):
        '''
        This will be implemented by sub-classes
        that inherit this class.
        '''
        
        raise NotImplementedError
