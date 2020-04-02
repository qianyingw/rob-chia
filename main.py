#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:25:09 2020

@author: qwang
"""

import pandas as pd
from tqdm import tqdm

from gensim import models
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
 
import os
os.chdir('/home/qwang/rob/rob-chia')
from data_helper import load_split_data
from vectorizer import bow, avg_w2v#, d2v
from model import sgd#, rf, xgboost

# data dir
os.chdir('/media/mynewdrive/rob/')
pkl_path = "data/rob_info_a.pkl"


#%% 
def train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid):
    """Output validation score dictionary"""
    clf.fit(X_train_vec, Y_train) 
    y_true, y_pred = Y_valid, clf.predict(X_valid_vec)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
    return {'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'rec': recall_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred),
            'spec': tn / (tn+fp)}


#%% Load data
rob_item="RandomizationTreatmentControl"
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_split_data(pkl_path, rob_item)

#['RandomizationTreatmentControl',
# 'BlindedOutcomeAssessment',
# 'SampleSizeCalculation',
# 'AnimalExclusions',
# 'AllocationConcealment',
# 'AnimalWelfareRegulations',
# 'ConflictsOfInterest']

#%% SGD + BOW
pars = {'ngram': [(2,2)],
        'min_df': [20],
        'max_feature': [1000],
        'tol': [0.001],
        'alpha': [0.001],    
        'tfidf': [True]}


pars = {'ngram': [(1,1), (2,2), (3,3), (1,2), (2,3), (1,3)],
        'min_df': [10, 20, 50, 100],
        'max_feature': [500, 1000, 2000, 5000],
        'tol': [0.001],
        'alpha': [0.001],    
        'tfidf': [True, False]}


res_ls = []
for p in tqdm(list(ParameterGrid(pars))):
    # Vectorization
    X_train_vec, X_valid_vec = bow(X_train, X_valid, p['ngram'], p['min_df'], p['max_feature'], p['tfidf']) 
    # Define classifier
    clf = sgd(p['tol'], p['alpha'])     
    try: 
        res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid) 
        p['m_iter'] = clf.n_iter_
        p.update(res_dict)  # Add score dict to pars dict
        res_ls.append(p)
    except:
        p.update({'m_iter': '', 'acc': '', 'f1': '', 'rec': '', 'prec': '', 'spec': ''})  # Add score dict to pars dict
        res_ls.append(p)
res_df = pd.DataFrame(res_ls) 


res_df.to_csv('sgd_bow_'+rob_item[:4].lower()+'.csv', sep=',', encoding='utf-8')


#%% SGD + Average(Word2Vec)
# Load word vectors
w2v = models.KeyedVectors.load_word2vec_format('wordvec/PubMed-and-PMC-w2v.bin', binary=True)  # w2v.vectors.shape

pars = {'min_df': [10],
        'max_feature': [5000],
        'tol': [0.001],
        'alpha': [0.5e-9, 1e-9, 0.5e-8, 1e-8, 0.5e-7, 1e-7, 0.5e-6, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2]}

res_ls = []
for p in tqdm(list(ParameterGrid(pars))):
    # Vectorization
    X_train_vec, X_valid_vec = avg_w2v(w2v, X_train, X_valid, p['min_df'], p['max_feature'])
    # Define classifier
    clf = sgd(p['tol'], p['alpha'])     
    try: 
        res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid)  
        p['m_iter'] = clf.n_iter_
        p.update(res_dict)  # Add score dict to pars dict
        res_ls.append(p)
    except:
        p.update({'m_iter': '', 'acc': '', 'f1': '', 'rec': '', 'prec': '', 'spec': ''})  # Add score dict to pars dict
        res_ls.append(p)
res_df = pd.DataFrame(res_ls) 


res_df.to_csv('sgd_w2v_'+rob_item[:4].lower()+'.csv', sep=',', encoding='utf-8')

#del w2v



