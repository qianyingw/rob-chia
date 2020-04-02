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


#%% Data processing
#pkl_path = 'data/rob_info_a.pkl'
#rob_item = "RandomizationTreatmentControl"

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


def load_split_data(pkl_path, rob_item):
    
    # Load data from pkl info
    df = pd.read_pickle(pkl_path)  # From rob_kiwi/data_helper.py
    # Split info data
    train_df = df[df.partition == "train"]
    valid_df = df[df.partition == "valid"] 
    test_df = df[df.partition == "test"] 
    # Obtain plain texts
    X_train = train_df.apply(read_text, axis=1)   
    X_valid = valid_df.apply(read_text, axis=1)     
    X_test = test_df.apply(read_text, axis=1) 
    # Obtain label for rob_item
    Y_train = train_df[rob_item]
    Y_valid = valid_df[rob_item]
    Y_test = test_df[rob_item]
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


