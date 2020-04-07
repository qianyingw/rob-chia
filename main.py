#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:25:09 2020

@author: qwang
"""

import pandas as pd
from tqdm import tqdm
import time

from gensim import models
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
 
import os
os.chdir('/home/qwang/rob/rob-chia')
from data_helper import load_split_data
from vectorizer import bow, avg_w2v, d2v
from model import sgd, lreg, rf


pkl_path = "/media/mynewdrive/rob/data/rob_info_a.pkl"


#%% 
def train(clf, p_dict, X_train_vec, Y_train, X_valid_vec, Y_valid):
    """Output validation score and pars dictionary"""
    try: 
        clf.fit(X_train_vec, Y_train) 
        y_true, y_pred = Y_valid, clf.predict(X_valid_vec)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
        res_dict = {'acc': accuracy_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred),
                    'rec': recall_score(y_true, y_pred),
                    'prec': precision_score(y_true, y_pred),
                    'spec': tn / (tn+fp)} 
        if clf.n_iter_:
            p_dict['m_iter'] = clf.n_iter_
        p_dict.update(res_dict)  # Add score dict to pars dict
    except:
        p_dict.update({'m_iter': '', 'acc': '', 'f1': '', 'rec': '', 'prec': '', 'spec': ''})  # Add score dict to pars dict    
    return p_dict


#%% Load data
rob_item = "RandomizationTreatmentControl"
use_d2v = True
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_split_data(pkl_path, rob_item, use_d2v)

#['RandomizationTreatmentControl',
# 'BlindedOutcomeAssessment',
# 'SampleSizeCalculation',
# 'AnimalExclusions',
# 'AllocationConcealment',
# 'AnimalWelfareRegulations',
# 'ConflictsOfInterest']

#%% clfs + bow
pars_vec = {'ngram': [(1,1), (2,2), (3,3), (1,2), (2,3), (1,3)],
            'min_df': [10, 20, 50, 100],
            'max_feature': [500, 1000, 2000, 5000],
            'tfidf': [True, False]}

pars_sgd = {'tol': [0.001], 'alpha': [0.001]}
pars_lr = {'C': [1, 100, 1000]}
pars_rf = {'n_tree': [100, 500, 1000], 'n_feature': ['sqrt', 'log2', None]}


res_sgd = []
res_lr = []
res_rf = []

for pv in tqdm(list(ParameterGrid(pars_vec))):
    # Vectorization
    X_train_vec, X_valid_vec = bow(X_train, X_valid, pv['ngram'], pv['min_df'], pv['max_feature'], pv['tfidf'])
    
    # Train sgd
    for p_sgd in list(ParameterGrid(pars_sgd)):       
        clf = sgd(p_sgd['tol'], p_sgd['alpha'])  
        p = {**pv, **p_sgd}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_sgd.append(p_score)
        
    # Train logistic regression
    for p_lr in list(ParameterGrid(pars_lr)):       
        clf = lreg(p_lr['C'])  
        p = {**pv, **p_lr}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid)  
        res_lr.append(p_score)
        
    # Train random forest
    for p_rf in list(ParameterGrid(pars_rf)):       
        clf = rf(p_rf['n_tree'], p_rf['n_feature'])  
        p = {**pv, **p_rf}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_rf.append(p_score)
        
# Convert result list to dataframe    
df_sgd = pd.DataFrame(res_sgd) 
df_lr = pd.DataFrame(res_lr) 
df_rf = pd.DataFrame(res_rf) 

# Output
df_sgd.to_csv('sgd_bow_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_lr.to_csv('lr_bow_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_rf.to_csv('rf_bow_'+rob_item+'.csv', sep=',', encoding='utf-8')

#%% clfs + avg_w2v
# Load word vectors
w2v = models.KeyedVectors.load_word2vec_format('wordvec/PubMed-and-PMC-w2v.bin', binary=True)  # w2v.vectors.shape

pars_vec = {'min_df': [10],
            'max_feature': [5000]}

pars_sgd = {'tol': [0.001], 'alpha': [0.5e-9, 1e-9, 0.5e-8, 1e-8, 0.5e-7, 1e-7, 0.5e-6, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2]}
pars_lr = {'C': [1, 100, 1000]}
pars_rf = {'n_tree': [100, 500, 1000], 'n_feature': ['sqrt', 'log2', None]}


res_sgd = []
res_lr = []
res_rf = []

for pv in tqdm(list(ParameterGrid(pars_vec))):
    # Vectorization
    X_train_vec, X_valid_vec = avg_w2v(w2v, X_train, X_valid, pv['min_df'], pv['max_feature'])
    
    # Train sgd
    for p_sgd in list(ParameterGrid(pars_sgd)):       
        clf = sgd(p_sgd['tol'], p_sgd['alpha'])  
        p = {**pv, **p_sgd}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_sgd.append(p_score)
        
    # Train logistic regression
    for p_lr in list(ParameterGrid(pars_lr)):       
        clf = lreg(p_lr['C'])  
        p = {**pv, **p_lr}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid)  
        res_lr.append(p_score)
        
    # Train random forest
    for p_rf in list(ParameterGrid(pars_rf)):       
        clf = rf(p_rf['n_tree'], p_rf['n_feature'])  
        p = {**pv, **p_rf}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_rf.append(p_score)
        
# Convert result list to dataframe    
df_sgd = pd.DataFrame(res_sgd) 
df_lr = pd.DataFrame(res_lr) 
df_rf = pd.DataFrame(res_rf) 

# Output
df_sgd.to_csv('sgd_w2v_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_lr.to_csv('lr_w2v_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_rf.to_csv('rf_w2v_'+rob_item+'.csv', sep=',', encoding='utf-8')


#%% clfs + d2v
pars_vec = {'d2v_model': ['dm', 'dbow'],
            'd2v_alpha': [0.01],
            'd2v_min_alpha': [0.001],
            'vector_size': [100, 200, 300, 400, 500],
            'min_count': [10],
            'max_vocab_size': [5000, 10000]}

pars_sgd = {'tol': [0.001], 'alpha': [1e-3]}
pars_lr = {'C': [1, 100, 1000]}
pars_rf = {'n_tree': [100, 500, 1000], 'n_feature': ['sqrt', 'log2', None]}

res_sgd = []
res_lr = []
res_rf = []

for pv in tqdm(list(ParameterGrid(pars_vec))):
    # Vectorization
    start = time.time()
    X_train_vec, Y_train, X_valid_vec, Y_valid = d2v(X_train, X_valid, pv['d2v_model'], pv['d2v_alpha'], pv['d2v_min_alpha'], 
                                                     pv['vector_size'], pv['min_count'], pv['max_vocab_size'])  
    print("Doc2Vec done: {:.2f} mins".format((time.time()-start)/60))
    
    # Train sgd
    for p_sgd in list(ParameterGrid(pars_sgd)):       
        clf = sgd(p_sgd['tol'], p_sgd['alpha'])  
        p = {**pv, **p_sgd}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_sgd.append(p_score)
        
    # Train logistic regression
    for p_lr in list(ParameterGrid(pars_lr)):       
        clf = lreg(p_lr['C'])  
        p = {**pv, **p_lr}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid)  
        res_lr.append(p_score)
        
    # Train random forest
    for p_rf in list(ParameterGrid(pars_rf)):       
        clf = rf(p_rf['n_tree'], p_rf['n_feature'])  
        p = {**pv, **p_rf}
        p_score = train(clf, p, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        res_rf.append(p_score)
        
# Convert result list to dataframe    
df_sgd = pd.DataFrame(res_sgd) 
df_lr = pd.DataFrame(res_lr) 
df_rf = pd.DataFrame(res_rf) 

# Output
df_sgd.to_csv('sgd_d2v_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_lr.to_csv('lr_d2v_'+rob_item+'.csv', sep=',', encoding='utf-8')
df_rf.to_csv('rf_d2v_'+rob_item+'.csv', sep=',', encoding='utf-8')