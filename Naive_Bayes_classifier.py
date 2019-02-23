# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:45:40 2019

@author: Justin Won
"""
import numpy as np
import pandas as pd
import operator

def preprocess(df):
    '''Remove c_charge_degree_M column due to redunduncy'''
    return df.drop(labels=['c_charge_degree_M'], axis=1)

def naive_bayes_train(df, features, target='two_year_recid', k=1):
    """returns class conditional probability distribution and class prior as dictionaries"""
    # initialize variables
    sample_size, feature_dim = df.shape[0], len(features)
    # print('sample_size, feature_dim: ', sample_size, feature_dim)
    y_count = {} # {key = label : value = count of label in sample},
    x_count = {} # dict of dicts of dicts = {key=label:value={key=feature:value={key=feature_val:value=count}}}
    feature_given_class = {} # Pr[X=x|Y=y]
    class_prior = {} # Pr[Y=y]
    
    # initialize dictionary keys
    for label in df[target].unique():
        y_count[label], class_prior[label] = 0, 0
        x_count[label], feature_given_class[label] = {}, {}
        for feature in features:
            x_count[label][feature], feature_given_class[label][feature] = {}, {}
    
    # print('y_count: ', y_count)
    # print('x_count: ', x_count)
    
    # update dictionary values
    for index, row in df.iterrows():
        label = row[target]
        y_count[label] += 1
        for feature in features:
            feature_val = row[feature]
            if feature_val in x_count[label][feature].keys():
                x_count[label][feature][feature_val] += 1
            else:
                x_count[label][feature][feature_val] = 1
    
    # print('y_count: ', y_count)
    # print('x_count: ', x_count)
    
    # find class prior probabilities
    for label in class_prior.keys():
        class_prior[label] = y_count[label]/sample_size
    
    # find feature given class probabilities with additive smoothing
    for label in feature_given_class.keys():
        for feature in feature_given_class[label].keys():
            for feature_val in x_count[label][feature].keys():
                feature_given_class[label][feature][feature_val] = \
                (x_count[label][feature][feature_val] + k) / (y_count[label] + k * feature_dim)
    # print('class prior: ', class_prior)
    # print('f|c: ', feature_given_class)
    return class_prior, feature_given_class, y_count

def predict(series, features, class_prior, feature_given_class, y_count, k=1):
    """given a series, return a belief distribution over possible labels"""
    belief = {}
    for label in class_prior.keys():
        prob = class_prior[label]
        for feature in features:
            feature_val = series[feature]
            # print(feature, feature_val)
            if feature_val in feature_given_class[label][feature]:
                prob *= feature_given_class[label][feature][feature_val]
            else:
                prob *= k/(y_count[label] + k * len(features))
        belief[label] = prob
    return belief

def evaluate(df, features, class_prior, feature_given_class, y_count, target = 'two_year_recid'):
    total_pred, accurate_pred = 0, 0
    y_preds = []
    for index, row in df.iterrows():
        y = row[target]
        belief = predict(row, features, class_prior, feature_given_class, y_count)
        y_pred = max(belief.items(), key=operator.itemgetter(1))[0]
        total_pred += 1
        if y_pred == y:
            accurate_pred += 1
        y_preds.append(y_pred)
    y_pred_df = pd.DataFrame({'prediction': y_preds})
    df = pd.concat([df, y_pred_df], axis=1)
    accuracy = accurate_pred/total_pred
    return accuracy, df

def perform_naive_bayes(train, test):
    train = preprocess(train)
    test = preprocess(test)
    features = list(train)[1:]
    class_prior, feature_given_class, y_count = naive_bayes_train(train, features)
    return evaluate(test, features, class_prior, feature_given_class, y_count)

if __name__ == '__main__':
    train = pd.read_csv('hw1data/propublicaTrain.csv')
    test = pd.read_csv('hw1data/propublicaTest.csv')
    
    print(perform_naive_bayes(train, test))