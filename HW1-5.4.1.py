# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:16:57 2019

@author: Justin Won
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def preprocess(df):
    '''Remove c_charge_degree_M column due to redunduncy'''
    return df.drop(labels=['c_charge_degree_M'], axis=1)

def split_categorical_and_quant(df):
    categorical_cols = ['two_year_recid', 'sex', 'race', 'c_charge_degree_F']
    # Need to include 'two_year_recid' in both because it is the target y
    quantitative_cols = ['two_year_recid', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    return df[categorical_cols], df[quantitative_cols]

def split_by_y(df):
    return df.loc[df['two_year_recid'] == 0, :], df.loc[df['two_year_recid'] == 1, :]

def split_y(df):
    X = df.loc[:, df.columns != 'two_year_recid']
    y = df['two_year_recid']
    return X, y

def estimate_mu(quant_df):
    '''Estimates the mean vector (mu) using discrete datapoints'''
    return np.array([quant_df[col].mean() for col in quant_df.columns])

def estimate_sigma(quant_X, mu):
    '''
    Estimates the covariance matrix (sigma) using discrete datapoints
    See https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/covariance/empirical_covariance_.py#L50
    '''
    X = np.asarray(quant_X)
    X = np.array([row - mu for row in X])
    # May need to do row by row substraction from X and then perform matrix multiplication
    sigma = X.T @ X / X.shape[0]
    return sigma

def fit_gaussian(quant_X):
    '''
    return mu, and sigma
    '''
    mu = estimate_mu(quant_X)
    sigma = estimate_sigma(quant_X, mu)
    return mu, sigma

def build_repr(row):
    key = ''
    if row['sex']:
        key += '1'
    else:
        key += '0'
    if row['race']:
        key += '1'
    else:
        key += '0'
    if row['c_charge_degree_F']:
        key += '1'
    else:
        key += '0'
    return key

def calculate_cat_pr(df):
    '''
    Take df and return dictionary that maps key (indexes) and
    values (probabilities).
    '''
    n = df.shape[0]
    try:
        categ_X, _ = split_y(df)
    except KeyError:
        pass
    keys = ['000', '001', '010', '011', '100', '101', '110', '111']
    values = [0] * 8
    pr_dict = dict(zip(keys, values))
    for index, row in df.iterrows():
        key = build_repr(row)
        pr_dict[key] += 1
        
    for key in pr_dict:
        pr_dict[key] /= n
    return pr_dict

def get_cat_pr(x):
    '''
    Take index of a row and return correct P_3 or P_4 based on the row's
    categorical columns' values.
    '''
    key = build_repr(x)
    categ_pr = None
    if x['two_year_recid'] == 0:
        categ_pr = cat_0_pr_dict[key]
    else:
        categ_pr = cat_1_pr_dict[key]
    return categ_pr

def predict(row, dist_0, dist_1, cat_0_pr_dict, cat_1_pr_dict, Pr_y_eq_0, Pr_y_eq_1):
    prediction = None
    prob = 0
    # Preprocessing to fit the row into the model
    categ_row, quant_row = split_categorical_and_quant(row)
    quant_row = quant_row[1:]
    categ_row = categ_row[1:]
    
    prob_0 = dist_0.pdf(quant_row)
    prob_0 *= cat_0_pr_dict[build_repr(categ_row)]
    prob_0 *= Pr_y_eq_0
    
    prob_1 = dist_1.pdf(quant_row)
    prob_1 *= cat_1_pr_dict[build_repr(categ_row)]
    prob_1 *= Pr_y_eq_1
    
    if prob_0 >= prob_1:
        prediction = 0
    else:
        prediction = 1
    return prediction

def evaluate(test_df, dist_0, dist_1, cat_0_pr_dict, cat_1_pr_dict, Pr_y_eq_0, Pr_y_eq_1):
    labels = test_df['two_year_recid']
    total = test_df.shape[0]
    predictions = []
    num_correct = 0
    for index, row in test_df.iterrows():
        prediction = predict(row, dist_0, dist_1, cat_0_pr_dict, cat_1_pr_dict, Pr_y_eq_0, Pr_y_eq_1)
        predictions.append(int(prediction))
        if prediction == labels[index]:
            num_correct += 1
    accuracy = num_correct / total
    return accuracy, predictions

def perform_mle(train_df, test_df):
    train = pd.read_csv('hw1data/propublicaTrain.csv')
    test = pd.read_csv('hw1data/propublicaTest.csv')
    
    # Preprocessing
    train_df = preprocess(train)
    test_df = preprocess(test)
    train_df.head()
    
    # Split to categorical and quantitative
    categ_train_df, quant_train_df = split_categorical_and_quant(train_df)
    categ_test_df, quant_test_df = split_categorical_and_quant(test_df)
    
    # Split the data into y = 0 and y = 1
    categ_0_train_df, categ_1_train_df = split_by_y(categ_train_df)
    quant_0_train_df, quant_1_train_df = split_by_y(quant_train_df)
    categ_0_test_df, categ_1_test_df = split_by_y(categ_test_df)
    quant_0_test_df, quant_1_test_df = split_by_y(quant_test_df)
    
    # Split the label from features
    quant_0_train_X, quant_0_train_y = split_y(quant_0_train_df)
    quant_1_train_X, quant_1_train_y = split_y(quant_1_train_df)
    
    mu0 = estimate_mu(quant_0_train_X)
    mu1 = estimate_mu(quant_1_train_X)
    
    sigma0 = estimate_sigma(quant_0_train_X, mu0)
    sigma1 = estimate_sigma(quant_1_train_X, mu1)
    
    mu0, simga0 = fit_gaussian(quant_0_train_X)
    mu1, simga1 = fit_gaussian(quant_1_train_X)
    
    dist_0 = multivariate_normal(mu0, sigma0)
    dist_1 = multivariate_normal(mu1, sigma1)
    
    cat_0_pr_dict = calculate_cat_pr(categ_0_train_df)
    cat_1_pr_dict = calculate_cat_pr(categ_1_train_df)
    
    # Probability that y = 0 and y = 1
    Pr_y_eq_0 = categ_0_train_df.shape[0] / train_df.shape[0]
    Pr_y_eq_1 = categ_1_train_df.shape[0] / train_df.shape[0]
    
    # Predict and evaluate
    accuracy, predictions = evaluate(test_df, dist_0, dist_1, cat_0_pr_dict, cat_1_pr_dict, Pr_y_eq_0, Pr_y_eq_1)
    predictions = pd.DataFrame({'prediction':predictions})
    test_df = pd.concat([test_df, predictions], axis=1)
    return accuracy, test_df

if __name__ == '__main__':
    train = pd.read_csv('hw1data/propublicaTrain.csv')
    test = pd.read_csv('hw1data/propublicaTest.csv')
    
    accuracy, test_df = perform_mle(train, test)
    print(test_df)
    