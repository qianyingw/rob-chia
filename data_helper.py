#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:35:44 2020

@author: qwang
"""

import pickle
import pandas as pd

import os
os.chdir('/media/mynewdrive/rob/')

from gensim.models.doc2vec import TaggedDocument



#pkl_path = 'data/rob_info_a.pkl'
#rob_item = "RandomizationTreatmentControl"

#%% Read text from each df row (for bow/avg_w2v) 
def read_text(row):
    pkl_path = os.path.join('data/rob_str', row['goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)        
    text = ''
    for l in sent_ls:
        t = ' '.join(l)
        text = text + t
    return text

#%% Read tokens from each df row (for d2v) 
def read_token(row):
    pkl_path = os.path.join('data/rob_str', row['goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)        
    tokens = []
    for l in sent_ls:
        tokens = tokens+ l[0].split(" ")
    return tokens


#%% 
 
def load_split_data(pkl_path, rob_item, use_d2v=False):
    # Load data from pkl info
    df = pd.read_pickle(pkl_path)  # From rob_kiwi/data_helper.py
    # Split info data
    train_df = df[df.partition == "train"]
    valid_df = df[df.partition == "valid"] 
    test_df = df[df.partition == "test"] 
    
    # Obtain plain texts for bow/avg_w2v
    X_train = train_df.apply(read_text, axis=1)   
    X_valid = valid_df.apply(read_text, axis=1)     
    X_test = test_df.apply(read_text, axis=1) 
    
    # Obtain tagged document for d2v
    if use_d2v == True:
        X_train = train_df.apply(lambda r: TaggedDocument(words=read_token(r), tags=[r[rob_item]]), axis=1)  # X_train.values[i]
        X_valid = valid_df.apply(lambda r: TaggedDocument(words=read_token(r), tags=[r[rob_item]]), axis=1)
        X_test = test_df.apply(lambda r: TaggedDocument(words=read_token(r), tags=[r[rob_item]]), axis=1)           

    # Obtain label for rob_item
    Y_train = train_df[rob_item]
    Y_valid = valid_df[rob_item]
    Y_test = test_df[rob_item]
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


