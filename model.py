#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:39:54 2020

@author: qwang
"""


from sklearn.linear_model import SGDClassifier


#from sklearn.linear_model import LogisticRegression
#from sklearn import utils

import os
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)



#%% SGD
def sgd(tol, alr):
    clf = SGDClassifier(loss='hinge', penalty='l2', verbose=0, class_weight='balanced', n_jobs=16, # n_samples / (output_dim * np.bincount(y))         
                        random_state=1234, shuffle=True, # Shuffle training data after each epoch        
                        max_iter=1000, early_stopping=True, validation_fraction=0.1, tol=tol, n_iter_no_change=5, # Early stopping
                        learning_rate='optimal', alpha=alr)
    return clf

#%% Random Forest
def rf():
    return

#%% Extreme Gradient Boosting
def xgboost():
    return


