#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:25:27 2020

@author: qwang
"""


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import Doc2Vec

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


#%% Doc2Vec
def d2v(train_tagged, valid_tagged, d2v_model, d2v_alpha, d2v_min_alpha, vector_size, min_count, max_vocab_size):
    
    if d2v_model == "dm":
        model = Doc2Vec(dm=1,  # 1: PV-DM. 0: PV-DBOW
                        dm_mean=1,  # 0: use the sum of the context word vectors. 1: use the mean
                        epochs=20, seed=1234, workers=1,
                        alpha=d2v_alpha, min_alpha=d2v_min_alpha,  # initial learning rate; linearly drops to min_alpha as training progresses                   
                        vector_size=vector_size,  # feature vector dim
                        min_count=min_count, max_vocab_size=max_vocab_size,
                        hs=0,  # 1: hierarchical softmax is used
                        negative=5)  # use negative sampling - how many “noise words” should be drawn
        
    if d2v_model == "dbow":
        model = Doc2Vec(dm=0,  # 1: PV-DM. 0: PV-DBOW
                        epochs=20, seed=1234, workers=1,
                        alpha=d2v_alpha, min_alpha=d2v_min_alpha,  # initial learning rate; linearly drops to min_alpha as training progresses                   
                        vector_size=vector_size,  # feature vector dim
                        min_count=min_count, max_vocab_size=max_vocab_size,
                        hs=0,  # 1: hierarchical softmax is used
                        negative=5)  # use negative sampling - how many “noise words” should be drawn                      
    
    model.build_vocab(train_tagged)
    model.train(train_tagged, total_examples=len(train_tagged.values), epochs=model.epochs)
    
    train_docs = train_tagged.values
    Y_train, X_train_vec = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in train_docs])

    valid_docs = valid_tagged.values
    Y_valid, X_valid_vec = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in valid_docs])
    
    return X_train_vec, Y_train, X_valid_vec, Y_valid

###
#    p = {'alpha': 0.001,
#  'd2v_alpha': 0.01,
#  'd2v_min_alpha': 0.001,
#  'd2v_model': 'dbow',
#  'max_vocab_size': 5000,
#  'min_count': 10,
#  'tol': 0.001,
#  'vector_size': 300}
#
#
#d2v_alpha = 0.01; d2v_min_alpha = 0.001; vector_size = 300; min_count=10;max_vocab_size=5000
#train_tagged = X_train; valid_tagged = X_valid
#
#X_train_vec, X_valid_vec = d2v(X_train, X_valid, 'dm', 0.01, 0.001, 300, 10, 5000)