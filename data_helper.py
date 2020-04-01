#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:35:44 2020

@author: qwang
"""

import pickle
import pandas as pd
import os
from tqdm import tqdm
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

import multiprocessing
cores = multiprocessing.cpu_count()  # 28

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec


from sklearn.linear_model import LogisticRegression
from sklearn import utils
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

#%% Data processing
df = pd.read_pickle('data/rob_info_a.pkl')  # From rob_kiwi/data_helper.py

# Obtain pre-splitted data
train_df = df[df.partition == "train"]
valid_df = df[df.partition == "valid"] 
test_df = df[df.partition == "test"] 

## Convert sentence lists of one doc to plain text
#def pkl_list_to_str(pkl_dir, pkl_name):
#    pkl_path = os.path.join(pkl_dir, pkl_name) 
#    with open(pkl_path, 'rb') as fin:
#        sent_ls = pickle.load(fin)        
#    text = ''
#    for l in sent_ls:
#        t = ' '.join(l)
#        text = text + t 
#    return text
 
        
#%% Format input for BOW
def read_text(row):
    pkl_path = os.path.join('data/rob_str', row['goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)        
    text = ''
    for l in sent_ls:
        t = ' '.join(l)
        text = text + t
    return text


rob_item = "RandomizationTreatmentControl"
train_df['text'] = train_df.apply(read_text, axis=1)        
valid_df['text'] = valid_df.apply(read_text, axis=1)     
test_df['text'] = test_df.apply(read_text, axis=1) 

#%% Format input for Doc2Vec (a document along is represented with a tag)
def pkl_list_to_token(pkl_dir, pkl_name):
    pkl_path = os.path.join(pkl_dir, pkl_name) 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)        
    tokens = []
    for l in sent_ls:
        tokens = tokens+ l[0].split(" ")
    return tokens


rob_item = "RandomizationTreatmentControl"
train_tagged = train_df.apply(lambda r: TaggedDocument(words=pkl_list_to_token(pkl_dir='data/rob_str', pkl_name=r['goldID']+'.pkl'), tags=[r[rob_item]]), axis=1)        
valid_tagged = valid_df.apply(lambda r: TaggedDocument(words=pkl_list_to_token(pkl_dir='data/rob_str', pkl_name=r['goldID']+'.pkl'), tags=[r[rob_item]]), axis=1)         



#%% Distributed Bag of Words (DBOW)
# Build vocabulary
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=4)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

# Train a doc2vec model
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


# Build feature vectors
def feature_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = feature_vectors(model_dbow, train_tagged)
y_valid, X_valid = feature_vectors(model_dbow, valid_tagged)


# Train Logistic Regression Classifier
logreg = LogisticRegression(n_jobs=1, C=1e4, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_valid)


print('Valid accuracy: {}%'.format(round(accuracy_score(y_valid, y_pred)*100,2)))
print('Valid F1 score: {}%'.format(round(f1_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid recall: {}%'.format(round(recall_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid precision: {}%'.format(round(precision_score(y_valid, y_pred, average='weighted')*100,2)))

#%% Distributed Memory (DM)
# Build vocabulary
model_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=4, alpha=0.065, min_alpha=0.065)
model_dm.build_vocab([x for x in tqdm(train_tagged.values)])

# Train a doc2vec model
for epoch in range(30):
    model_dm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dm.alpha -= 0.002
    model_dm.min_alpha = model_dm.alpha

# Build feature vectors
y_train, X_train = feature_vectors(model_dbow, train_tagged)
y_valid, X_valid = feature_vectors(model_dbow, valid_tagged)


# Train Logistic Regression Classifier
logreg = LogisticRegression(n_jobs=1, C=1e4, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_valid)


print('Valid accuracy: {}%'.format(round(accuracy_score(y_valid, y_pred)*100,2)))
print('Valid F1 score: {}%'.format(round(f1_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid recall: {}%'.format(round(recall_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid precision: {}%'.format(round(precision_score(y_valid, y_pred, average='weighted')*100,2)))

#%% Concatenate DBOW and DM
# Delete temporary training data to free up RAM
model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

# Concatenate two models
new_model = ConcatenatedDoc2Vec([model_dbow, model_dm])

# Build feature vectors
y_train, X_train = feature_vectors(model_dbow, train_tagged)
y_valid, X_valid = feature_vectors(model_dbow, valid_tagged)


# Train Logistic Regression Classifier
logreg = LogisticRegression(n_jobs=1, C=1e4, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_valid)


print('Valid accuracy: {}%'.format(round(accuracy_score(y_valid, y_pred)*100,2)))
print('Valid F1 score: {}%'.format(round(f1_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid recall: {}%'.format(round(recall_score(y_valid, y_pred, average='weighted')*100,2)))
print('Valid precision: {}%'.format(round(precision_score(y_valid, y_pred, average='weighted')*100,2)))