#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:25:09 2020

@author: qwang
"""
import pandas as pd
import pickle
import itertools
from tqdm import tqdm

from gensim import models


import os
src_dir = '/home/qwang/rob/rob-chia'
os.chdir(src_dir)
from model import bow, avg_w2v, sgd, train
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)



#%% Data processing
df = pd.read_pickle('data/rob_info_a.pkl')  # From rob_kiwi/data_helper.py
     
# Format input for BOW
def read_text(row):
    pkl_path = os.path.join('data/rob_str', row['goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)        
    text = ''
    for l in sent_ls:
        t = ' '.join(l)
        text = text + t
    return text


# Obtain pre-splitted data
train_df = df[df.partition == "train"]
valid_df = df[df.partition == "valid"] 
test_df = df[df.partition == "test"] 

X_train = train_df.apply(read_text, axis=1)   
X_valid = valid_df.apply(read_text, axis=1)     
X_test = test_df.apply(read_text, axis=1) 


#%% SGD + BOW
ngrams = ((1,1), (1,2), (2,2), (1,3), (1,4))  
dfs = (5, 10, 20, 50, 100)
feats = (1000, 5000, 10000)
alpha = (0.001, 0.01, 0.1, 1)
tols = (0.001,)
idfs = (True, False)

## Random ##
rob_item = "RandomizationTreatmentControl"
Y_train = train_df[rob_item]
Y_valid = valid_df[rob_item]
Y_test = test_df[rob_item]


res_df = pd.DataFrame(columns=['ngram','min_df','max_feature','tol','alpha','tfidf','n_iter','acc','f1','rec','prec','spec'])  
for ng, md, mf, alr, t, idf in tqdm(itertools.product(ngrams, dfs, feats, alpha, tols, idfs)):
    # Prepare data
    X_train_vec, X_valid_vec = bow(X_train, X_valid, ng, md, mf, idf)
    # Define classifier
    clf = sgd(t, alr)    
    # Train
    res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid)  
    res_df = res_df.append({'ngram': ng, 'min_df': md, 'max_feature': mf, 'tol': t, 'alpha': alr, 'tfidf': idf, 'n_iter': clf.n_iter_,
                            'acc': res_dict['acc'],
                            'f1': res_dict['f1'],
                            'rec': res_dict['rec'],
                            'prec': res_dict['prec'],
                            'spec': res_dict['spec']}, ignore_index=True)     
        
res_df.to_csv('sgd_bow_r.csv', sep=',', encoding='utf-8')

## Blind ##
rob_item = "BlindedOutcomeAssessment"
Y_train = train_df[rob_item]
Y_valid = valid_df[rob_item]
Y_test = test_df[rob_item]


res_df = pd.DataFrame(columns=['ngram','min_df','max_feature','tol','alpha','tfidf','n_iter','acc','f1','rec','prec','spec'])  
for ng, md, mf, alr, t, idf in tqdm(itertools.product(ngrams, dfs, feats, alpha, tols, idfs)):
    # Prepare data
    X_train_vec, X_valid_vec = bow(X_train, X_valid, ng, md, mf, idf)
    # Define classifier
    clf = sgd(t, alr)    
    # Train
    res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid)  
    res_df = res_df.append({'ngram': ng, 'min_df': md, 'max_feature': mf, 'tol': t, 'alpha': alr, 'tfidf': idf, 'n_iter': clf.n_iter_,
                            'acc': res_dict['acc'],
                            'f1': res_dict['f1'],
                            'rec': res_dict['rec'],
                            'prec': res_dict['prec'],
                            'spec': res_dict['spec']}, ignore_index=True)     
        
res_df.to_csv('sgd_bow_b.csv', sep=',', encoding='utf-8')

## Interest ##
rob_item = "ConflictsOfInterest"
Y_train = train_df[rob_item]
Y_valid = valid_df[rob_item]
Y_test = test_df[rob_item]


res_df = pd.DataFrame(columns=['ngram','min_df','max_feature','tol','alpha','tfidf','n_iter','acc','f1','rec','prec','spec'])  
for ng, md, mf, alr, t, idf in tqdm(itertools.product(ngrams, dfs, feats, alpha, tols, idfs)):
    # Prepare data
    X_train_vec, X_valid_vec = bow(X_train, X_valid, ng, md, mf, idf)
    # Define classifier
    clf = sgd(t, alr)    
    # Train
    res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid)  
    res_df = res_df.append({'ngram': ng, 'min_df': md, 'max_feature': mf, 'tol': t, 'alpha': alr, 'tfidf': idf, 'n_iter': clf.n_iter_,
                            'acc': res_dict['acc'],
                            'f1': res_dict['f1'],
                            'rec': res_dict['rec'],
                            'prec': res_dict['prec'],
                            'spec': res_dict['spec']}, ignore_index=True)     
        
res_df.to_csv('sgd_bow_i.csv', sep=',', encoding='utf-8')

#%% SGD + Average(Word2Vec)

alpha = (0.001, 0.01, 0.1, 1)
tols = (0.001,)


dfs = (10,100)
feats = (1000, 9000)
alpha = (0.01,)
tols = (0.001,)


# Load word vectors
w2v = models.KeyedVectors.load_word2vec_format('wordvec/PubMed-and-PMC-w2v.bin', binary=True)  # w2v.vectors.shape

res_df = pd.DataFrame(columns=['min_df','max_feature','tol','alpha', 'n_iter','acc','f1','rec','prec','spec'])  
for md, mf, t, alr in tqdm(itertools.product(dfs, feats, tols, alpha)):
    # Prepare data
    X_train_vec, X_valid_vec = avg_w2v(w2v, X_train, X_valid, min_df=md, max_f=mf)
    # Define classifier
    clf = sgd(t, alr)    
    # Train
    res_dict = train(clf, X_train_vec, Y_train, X_valid_vec, Y_valid)  
    res_df = res_df.append({'min_df': md, 'max_feature': mf, 'tol': t, 'alpha': alr, 'n_iter': clf.n_iter_,
                            'acc': res_dict['acc'],
                            'f1': res_dict['f1'],
                            'rec': res_dict['rec'],
                            'prec': res_dict['prec'],
                            'spec': res_dict['spec']}, ignore_index=True)     
        
res_df.to_csv('sgd_w2v_r.csv', sep=',', encoding='utf-8')
      


