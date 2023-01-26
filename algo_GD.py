
import numpy as np

import algos


class GD_Optim(algos.LineSearch):
    '''
    Optimization of a known function via gradient descent.
    '''
    
    ## Clerical methods. ##

    def __init__(self, w_init, step, t_max=None, thres=None,
                 verbose=False, store=False, name="GD_Optim"):

        super(GD_Optim, self).__init__(w_init=w_init, step=step,
                                       t_max=t_max, thres=thres,
                                       verbose=verbose,
                                       store=store, name=name)
    
    ## Implementation of new direction computation. ##
    
    def newdir(self, model, data=None):

        return (-1) * model.g_opt(w=self.w)


class GD_ERM(algos.LineSearch):
    '''
    Empirical risk minimization via line search gradient descent.
    '''
    
    ## Clerical methods. ##

    def __init__(self, w_init, step, t_max=None, thres=None,
                 verbose=False, store=False, paras=None, name="GD_ERM"):

        super(GD_ERM, self).__init__(w_init=w_init, step=step,
                                     t_max=t_max, thres=thres,
                                     verbose=verbose,
                                     store=store, name=name)
        
        self.paras = paras
    
    
    ## Implementation of new direction computation. ##
    
    def newdir(self, model, data):

        return (-1) * np.mean(model.g_tr(w=self.w,data=data,paras=self.paras),
                              axis=0, keepdims=True)


class SGD_ERM(algos.LineSearch):
    '''
    Empirical risk minimization via line search gradient descent.
    '''
    
    ## Clerical methods. ##
    
    def __init__(self, w_init, step,
                 batchsize, replace,
                 reg_coef=None,
                 t_max=None, thres=None,
                 verbose=False, store=False, name="SGD_ERM"):
    
        super(SGD_ERM, self).__init__(w_init=w_init, step=step,
                                  t_max=t_max, thres=thres,
                                  verbose=verbose,
                                  store=store, name=name)
        
        self.batchsize = batchsize
        self.replace = replace
        if reg_coef is None:
            self.paras = None
        else:
            self.paras = {"reg_coef": reg_coef}
        
    
    ## Implementation of new direction computation. ##
    
    def newdir(self, model, data):
        
        idx_shuf = np.random.choice(data.n_tr,
                                    size=self.batchsize,
                                    replace=self.replace)
        
        return (-1) * np.mean(model.g_tr(w=self.w,
                                         data=data,
                                         n_idx=idx_shuf,
                                         paras=self.paras),
                              axis=0, keepdims=True)
        
        

    
