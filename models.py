
import numpy as np


class Model:
    '''
    Base class for model objects.
    '''

    ## Clerical methods. ##

    def __init__(self, name=""):
        
        self.name = name


    def __str__(self):
        
        out = "Model name: {}".format(self.name)
        return out


    ## Loss-related methods. ##
    
    def l_imp(self, w=None, X=None, y=None, paras=None):
        
        raise NotImplementedError

    
    def l_tr(self, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              paras=paras)
        else:
            return self.l_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              paras=paras)

        
    def l_te(self, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              paras=paras)
        else:
            return self.l_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              paras=paras)

    def g_imp(self, w=None, X=None, y=None, paras=None):
        
        raise NotImplementedError
    
    def g_tr(self, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              paras=paras)
        else:
            return self.g_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              paras=paras)
    
    def g_te(self, w, data, n_idx=None, paras=None):
        
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              paras=paras)
        else:
            return self.g_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              paras=paras)
    
    
class Regressor(Model):
    '''
    Generic regression model, an object with methods
    for both training and evaluating regressors.
    '''

    ## Clerical methods. ##
    
    def __init__(self, data=None, name=""):
        
        super(Regressor, self).__init__(name=name)
        
        # If provided data, record number of features.
        if data is not None:
            self.register(data=data)
        
        
    def register(self, data):

        self.numfeats = data.X_tr.shape[1]
        
        
    def __call__(self, X):

        return self._predict(X=X)

        
    def _predict(self, X):
        
        raise NotImplementedError("To be implemented by sub-classes.")

    

class Classifier(Model):
    '''
    Generic classifier model, an object with methods
    for both training and evaluating classifiers.
    '''

    ## Clerical methods. ##
    
    def __init__(self, data=None, name=""):
        
        super(Classifier, self).__init__(name=name)

        self.labels = None
        self.nc = None

        if data is not None:
            self.register(data=data)

            
    def register(self, data):

        self.labels, self.nc = self._get_labels(data=data)
        

    def __call__(self, X):

        return self._classify(X=X)


    def _get_labels(self, data):
        '''
        Get all the (unique) labels that appear in the data.
        Returns a tuple of form (labels, count), where
        - labels: array of shape (count,1)
        - count: the number of unique labels.
        '''
        
        cond_tr = data.y_tr is None
        cond_te = data.y_te is None
        
        if cond_tr and cond_te:
            raise ValueError("No label data provided!")
        else:
            if cond_tr:
                out_labels = np.unique(data.y_te)
            elif cond_te:
                out_labels = np.unique(data.y_tr)
            else:
                out_labels = np.unique(
                    np.concatenate((data.y_tr,data.y_te), axis=0)
                )
            count = out_labels.size
            return (out_labels.reshape((count,1)), count)

        
    def _classify(self, X):
        
        raise NotImplementedError("To be implemented by sub-classes.")
    

    def _onehot(self, y):
        '''
        A function for encoding y into a one-hot vector.
        Inputs:
        - y is a (k,1) array, taking values in {0,1,...,nc-1}.
        Outputs:
        - A one-hot encoding of shape (k,nc).
        '''

        k, num_labels = y.shape

        if self.nc is None:
            raise TypeError("Number of classes must be specified.")

        if num_labels != 1:
            raise ValueError("Not built to deal with multi-label data.")
        
        nc = self.nc
        
        C = np.zeros((k,nc), dtype=y.dtype)

        for i in range(k):
            idx_hot = y[i,0]
            C[i,idx_hot] = 1

        return C
        

    ## Performance-related methods. ##

    def class_perf(self, y_est, y_true):
        '''
        Given class label estimates and true values,
        compute the fraction of correct classifications
        made for each label, yielding typical binary
        classification performance metrics.

        Input:
        y_est and y_true are (k x 1) matrices of labels.

        Output:
        Returns a dictionary with two components, (1) being
        the fraction of correctly classified labels, and
        (2) being a dict of per-label precison/recall/F1
        scores. 
        '''
        
        # First, get the classification rate.
        k = y_est.size
        num_correct = (y_est == y_true).sum()
        frac_correct = num_correct / k
        frac_incorrect = 1.0 - frac_correct

        # Then, get precision/recall for each class.
        prec_rec = { i:None for i in range(self.nc) } # initialize

        for c in range(self.nc):

            idx_c = (y_true == c)
            idx_notc = (idx_c == False)

            TP = (y_est[idx_c] == c).sum()
            FN = idx_c.sum() - TP
            FP = (y_est[idx_notc] == c).sum()
            TN = idx_notc.sum() - FP

            # Precision.
            if (TP == 0 and FP == 0):
                prec = 0
            else:
                prec = TP / (TP+FP)

            # Recall.
            if (TP == 0 and FN == 0):
                rec = 0
            else:
                rec = TP / (TP+FN)

            # F1 (harmonic mean of precision and recall).
            if (prec == 0 or rec == 0):
                f1 = 0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            prec_rec[c] = {"P": prec,
                           "R": rec,
                           "F1": f1}

        return {"rate": frac_incorrect,
                "PRF1": prec_rec}



class Function:
    '''
    Base class for models which are just fixed functions.
    '''

    ## Clerical methods. ##

    def __init__(self, name=""):
        
        self.name = name


    def __str__(self):
        
        out = "Model name: {}".format(self.name)
        return out
    
    
    def __call__(self, w):
        
        return self.f_opt(w=w)


    ## Computation-related methods. ##
    
    def f_opt(self, w=None):
        '''
        Returns the function value; implement in sub-classes.
        '''
        
        raise NotImplementedError
    
    
    def g_opt(self, w=None):
        '''
        Returns the gradient; implement in sub-classes.
        '''
        
        raise NotImplementedError


    
