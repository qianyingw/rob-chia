#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:39:54 2020

@author: qwang
"""


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#%% SGD
def sgd(tol, alr):
    clf = SGDClassifier(loss='hinge', penalty='l2', verbose=0, class_weight='balanced', n_jobs=16, # n_samples / (output_dim * np.bincount(y))         
                        random_state=1234, shuffle=True, # Shuffle training data after each epoch        
                        max_iter=1000, early_stopping=True, validation_fraction=0.1, tol=tol, n_iter_no_change=5, # Early stopping
                        learning_rate='optimal', alpha=alr)
    return clf


#%% Logistic Regression
def lreg(c):
    clf = LogisticRegression(penalty='l2', class_weight='balanced', n_jobs=4,
                             tol=1e-4, dual=False,  # Prefer dual=False when n_samples > n_features.                            
                             C=c,  # smaller value specify stronger regularization
                             random_state=1234, max_iter=1000,
                             solver='lbfgs')

#    clf = LogisticRegression(n_jobs=1, C=1e4, max_iter=1000)
    return clf


#%% Random Forest Classifier
def rf(n_tree, n_feature):
    clf = RandomForestClassifier(n_estimators=n_tree, max_features=n_feature, class_weight='balanced', 
                                 max_depth=2, random_state=1234)
    return clf

#%% Extreme Gradient Boosting
def xgboost():
    return


