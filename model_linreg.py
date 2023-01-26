
import numpy as np
from scipy import stats

import models
import legendre


class LinReg(models.Regressor):
    '''
    General-purpose linear regression base class.
    '''

    ## Clerical methods. ##

    def __init__(self, data=None, name="Linear regression"):

        super(LinReg, self).__init__(data=data, name=name)


    def __call__(self, X, w):
        
        return self._predict(X=X, w=w)
    

    ## Methods related to execution. ##
    
    def _predict(self, w, X):
        '''
        Predict real-valued response.
        w is a (d x 1) array of weights.
        X is a (k x d) matrix of k observations.
        Returns array of shape (k x 1) of predicted values.
        '''
        return X.dot(w)
    
    
    ## Methods related to feedback. ##
    
    def corr(self, w, X, y):
        '''
        Generic wrapper for Pearson's correlation coefficient,
        computed for the predicted response and the
        actual response.
        '''
        
        yhat = self._predict(w=w, X=X)
        
        if np.sum(np.abs(yhat)) < 0.0001:
            return 0.0
        else:
            return stats.pearsonr(yhat.flatten(), y.flatten())[0]
    
    
class LinReg_Ridge(LinReg):
    '''
    An orthodox linear regression model
    using squared error and squared l2-norm
    for regularization (if desired).
    '''
    
    ## Clerical methods. ##
    
    def __init__(self, data=None, name="Linear regression: Ridge"):
        
        super(LinReg_Ridge, self).__init__(data=data, name=name)
        
        self.d = self.numfeats
        
    
    ## Methods related to feedback. ##
    
    def l_imp(self, w, X, y, paras=None):
        '''
        Implementation of squared error under linear model
        for regression, with squared l2-norm penalty.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.
        lamreg is a regularization parameter (l2 penalty).

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        if paras is None:
            return (y-self._predict(w=w,X=X))**2/2
        else:
            lamreg = paras["lamreg"]
            penalty = lamreg * np.linalg.norm(w)**2 / X.shape[0] # note division by k.
            return (y-self._predict(w=w,X=X))**2/2 + penalty
        
        
    def g_imp(self, w, X, y, paras=None):
        '''
        Implementation of the gradient of squared error
        under a linear regression model, with squared
        l2-norm penalty.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.

        Output:
        A (k x numfeats) matrix of gradients evaluated
        at k points.
        '''
        if paras is None:
            return (y-self._predict(w=w,X=X))*(-1)*X
        else:
            lamreg = paras["lamreg"]
            penalty = lamreg*2*w.T / X.shape[0] # note division by k.
            return (y-self._predict(w=w,X=X))*(-1)*X + penalty


class LinReg_LASSO(LinReg):
    '''
    An orthodox linear regression model
    using squared error and the l1-norm
    for regularization (if desired).
    '''
    
    ## Clerical methods. ##
    
    def __init__(self, data=None, name="Linear regression: LASSO"):
        
        super(LinReg_LASSO, self).__init__(data=data, name=name)
        
        self.d = self.numfeats
        
    
    ## Methods related to feedback. ##
    
    def l_imp(self, w, X, y, paras=None):
        '''
        Implementation of squared error under linear model
        for regression, with l1-norm penalty.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.
        lamreg is a regularization parameter (l2 penalty).

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        if paras is None:
            return (y-self._predict(w=w,X=X))**2/2
        else:
            lamreg = paras["lamreg"]
            penalty = lamreg * np.linalg.norm(w, ord=1)
            return (y-self._predict(w=w,X=X))**2/2 + penalty
        
        
    def g_imp(self, w, X, y, paras=None):
        '''
        Implementation of the gradient of squared error
        under a linear regression model, with l1-norm penalty.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.

        Output:
        A (k x numfeats) matrix of gradients evaluated
        at k points.
        '''
        if paras is None:
            return (y-self._predict(w=w,X=X))*(-1)*X
        else:
            lamreg = paras["lamreg"]
            penalty = lamreg*np.sign(w.T)
            return (y-self._predict(w=w,X=X))*(-1)*X + penalty
    

    def g_j_imp(self, j, w, X, y, paras=None): ## NEW!
        '''
        Implementation of jth element of gradient.
        '''
        if paras is None:
            return (y-self._predict(w=w,X=X))*(-1)*np.take(a=X, indices=[j], axis=1)
        else:
            lamreg = paras["lamreg"]
            penalty = lamreg*np.sign(w[j,0])
            return (y-self._predict(w=w,X=X))*(-1)*np.take(a=X, indices=[j], axis=1) + penalty
    
    
    def g_j_tr(self, j, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.g_j_imp(j=j, w=w, X=data.X_tr,
                                y=data.y_tr,
                                paras=paras)
        else:
            return self.g_j_imp(j=j, w=w, X=data.X_tr[n_idx,:],
                                y=data.y_tr[n_idx,:],
                                paras=paras)
    
    def g_j_te(self, j, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.g_j_imp(j=j, w=w, X=data.X_te,
                                y=data.y_te,
                                paras=paras)
        else:
            return self.g_j_imp(j=j, w=w, X=data.X_te[n_idx,:],
                                y=data.y_te[n_idx,:],
                                paras=paras)

    
