#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:39:54 2020

@author: qwang
"""

import pickle
import pandas as pd
import os

data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

#from sklearn.linear_model import LogisticRegression
#from sklearn import utils

#%% Data processing
df = pd.read_pickle('data/rob_info_a.pkl')  # From rob_kiwi/data_helper.py
rob_item = "RandomizationTreatmentControl"
       
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

Y_train = train_df[rob_item]
Y_valid = valid_df[rob_item]
Y_test = test_df[rob_item]

#%%

#CountVectorizer(token_pattern = r'\w{1,}',
#                ngram_range = (1,1), # (1,2), (2,2)
#                min_df = 10,
#                max_feature = 5000)
#
#TfidfTransformer(norm = 'l2', # l2, l1, None
#                 use_idf = True, smooth_idf = True)
#
#
#SGDClassifier(loss = 'hinge', penalty = 'l2',  
#              class_weight = 'balanced',  # n_samples / (output_dim * np.bincount(y))
#              # Shuffle training data after each epoch 
#              random_state = 1234, shuffle = True, 
#              # Early stopping
#              max_iter = 1000, early_stopping = True, validation_fraction = 0.1, 
#              tol = 0.001, n_iter_no_change = 5,
#              # Learning rate
#              learning_rate = 'optimal', # 'adaptive'
#              alpha = 0.0001 # used to compute learning_rate when set to 'optimal'              
#              )



# %% Normal (non grid search cv)
    
pars = [1000, 2000]
   

report = pd.DataFrame(columns=['par', 'acc','f1','rec','prec','spec'])   
      
for p in pars:    
     
    ## CountVectorizer ##  
    count_vec = CountVectorizer(ngram_range=(1,1), min_df=10, max_features=p, token_pattern=r'\w{1,}')  # (1,2), (2,2)
    X_train_count = count_vec.fit_transform(X_train)
    X_valid_count = count_vec.fit_transform(X_valid)
    # X_test_count = count_vect.fit_transform(X_test)
    
    ## TfidfTransformer ## 
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)  # l2, l1, None
    X_train_tfidf = tfidf.fit_transform(X_train_count)
    X_valid_tfidf = tfidf.fit_transform(X_valid_count)
    # X_test_tfidf = tfidf.fit_transform(X_test_count)
    
    ## SGDClassifier ##
    sgd = SGDClassifier(loss='hinge', penalty='l2', class_weight = 'balanced',  # n_samples / (output_dim * np.bincount(y))         
                        random_state=1234, shuffle=True,  # Shuffle training data after each epoch        
                        max_iter=1000, early_stopping=True, validation_fraction=0.1, tol=0.001, n_iter_no_change=5, # Early stopping
                        learning_rate='optimal', alpha=0.0001) # used to compute learning_rate when set to 'optimal' # 'adaptive'
    sgd.fit(X_train_tfidf, Y_train)    


    ## Validation ##
    Y_valid_predict = sgd.predict(X_valid_tfidf)
    y_true, y_pred = Y_valid, Y_valid_predict
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
    report = report.append({'par': p,
                            'acc': accuracy_score(y_true, y_pred),
                            'f1': f1_score(y_true, y_pred),
                            'rec': recall_score(y_true, y_pred),
                            'prec': precision_score(y_true, y_pred),
                            'spec': tn / (tn+fp)}, ignore_index=True)
    ## Test ##
    #y_test_predict = sgd.predict(X_test_tfidf)     
    #y_true, y_pred = y_test, y_test_predict
    
print(report)



