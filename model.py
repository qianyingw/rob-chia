#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:39:54 2020

@author: qwang
"""

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

#from sklearn.linear_model import LogisticRegression
#from sklearn import utils

import os
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)


#%% Bag-of-Words with/without TFIDF
def bow(X_train, X_valid, ngram, min_df, max_f, use_tfidf):
    
    count_vec = CountVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_f)  # (1,2), (2,2)
    X_train_vec = count_vec.fit_transform(X_train)
    X_valid_vec = count_vec.fit_transform(X_valid)    
    if use_tfidf == True:
        tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)  # l2, l1, None
        X_train_vec = tfidf.fit_transform(X_train_vec)
        X_valid_vec = tfidf.fit_transform(X_valid_vec)
    
    return X_train_vec, X_valid_vec

#%% Average Word2Vec
# Load word vectors
# w2v = models.KeyedVectors.load_word2vec_format('wordvec/PubMed-and-PMC-w2v.bin', binary=True)  # w2v.vectors.shape

def avg_w2v(w2v, X_train, X_valid, min_df, max_f): 
    # Build vocab from training data
    vect_w2v = CountVectorizer(min_df=min_df, max_features=max_f, vocabulary=w2v.index2word)
    vect_w2v.fit(X_train)
    
    X_train_docs = vect_w2v.inverse_transform(vect_w2v.transform(X_train))  # X_train_docs[i].shape = (doc_i_len,)
    X_valid_docs = vect_w2v.inverse_transform(vect_w2v.transform(X_valid))
    
    # Convert doc to embedding matrix
    # w2v[X_train_docs[i]].shape = (doc_i_len, 200)
    X_train_vec = np.vstack([np.mean(w2v[doc], axis=0) for doc in X_train_docs])  # X_train_vec.shape=(6272, 200)
    X_valid_vec = np.vstack([np.mean(w2v[doc], axis=0) for doc in X_valid_docs])  # X_valid_vec.shape=(784, 200)

    return X_train_vec, X_valid_vec    

#%% Classifiers
def sgd(tol, alr):
    clf = SGDClassifier(loss='hinge', penalty='l2', verbose=0, class_weight='balanced', n_jobs=16, # n_samples / (output_dim * np.bincount(y))         
                        random_state=1234, shuffle=True, # Shuffle training data after each epoch        
                        max_iter=1000, early_stopping=True, validation_fraction=0.1, tol=tol, n_iter_no_change=5, # Early stopping
                        learning_rate='optimal', alpha=alr)
    return clf

def random_forest():
    return

def xgboost():
    return

#%% Train
def train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid):
  
    clf.fit(X_train_vec, Y_train)
    y_true, y_pred = Y_valid, clf.predict(X_valid_vec)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
    
    return {'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'rec': recall_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred),
            'spec': tn / (tn+fp)}



