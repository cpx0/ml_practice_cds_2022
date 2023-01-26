
import numpy as np
import tables

import data
import model_perceptron
import helpers as hlp


_datafile = "vim-2_encoder_mjh.h5"


def num_voxels(part_id):
    '''
    To see how many voxels the specified
    participant has, check one of the
    response arrays (both tr/te same number).
    '''
    with tables.open_file(_datafile, mode="r") as f:
        return f.get_node(where="/sub"+str(part_id)+"/te", name="y").shape[0]
    

def gen(part_id, vox):
    '''
    Data generation function for the
    vim-2 encoder task, for participant
    specified by part_id in {1,2,3}.
    '''
    
    with tables.open_file(_datafile, mode="r") as f:
        X_tr = f.get_node(where=f.root.tr, name="X").read()
        X_te = f.get_node(where=f.root.te, name="X").read()
        
        n_tr = X_tr.shape[0]
        n_te = X_te.shape[0]
        
        node_y_tr = f.get_node(where="/sub"+str(part_id)+"/tr", name="y")
        node_y_te = f.get_node(where="/sub"+str(part_id)+"/te", name="y")
        
        y_tr = node_y_tr.read()[vox,:].reshape(n_tr,1)
        y_te = node_y_te.read()[vox,:].reshape(n_te,1)
    
    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="vim-2_encoder")


