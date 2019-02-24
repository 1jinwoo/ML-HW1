# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:35:47 2019

@author: Justin Won
"""
import numpy as np
import pandas as pd
import random
from scipy import stats

# Normed Distance
def distance(x,y,p):
    if (p=='inf'):
        all_dist = np.zeros(np.shape(x)[0])
        for i in range(np.shape(x)[0]):
            all_dist[i] = abs(x[i]-y[i])
        return np.amax(all_dist)
    else:
        sum = 0
        for i in range(np.shape(x)[0]):
            sum += (abs(x[i]-y[i]))**p
        return sum**(1/p)

def perform_knn(train, test):
    #Training-Validation Split
    raw_train_labels=np.array(train['two_year_recid'])
    raw_train_input = np.array([np.array(train['sex']),np.array(train['age']),
                                np.array(train['race']),np.array(train['juv_fel_count']),
                                np.array(train['juv_misd_count']),np.array(train['juv_other_count']),
                                np.array(train['priors_count']),np.array(train['c_charge_degree_F'])])
    validation_pct = 20
    val_size = validation_pct*len(raw_train_labels)//100
    val_ind = random.sample(range(len(raw_train_labels)), val_size)
    validation_input = np.zeros((8,val_size))
    validation_labels = np.zeros(val_size)
    training_input = np.zeros((8,len(raw_train_labels)-val_size))
    training_labels = np.zeros(len(raw_train_labels)-val_size)
    cat_names = ['sex','age','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree_F']
    val_cnt = 0
    trn_cnt = 0
    for i in range(len(raw_train_labels)):
        if i in val_ind:
            for j in range(len(cat_names)):
                validation_input[j,val_cnt]=train[cat_names[j]].iloc[i]
            validation_labels[val_cnt]=train['two_year_recid'].iloc[i]
            val_cnt += 1
        else:
            for j in range(len(cat_names)):
                training_input[j,trn_cnt]=train[cat_names[j]].iloc[i]
            training_labels[trn_cnt]=train['two_year_recid'].iloc[i]
            trn_cnt += 1
    
    #Reading in Test Data
    test_labels = np.array(test['two_year_recid'])
    test_input = np.array([np.array(test['sex']),np.array(test['age']),np.array(test['race']),
                           np.array(test['juv_fel_count']),np.array(test['juv_misd_count']),
                           np.array(test['juv_other_count']),np.array(test['priors_count']),
                           np.array(test['c_charge_degree_F'])])
    
    #Running the best model, based on validation data, on our testing data
    k_final=15
    p_final=1
    pred_test_labels = np.zeros(len(test_labels))
    for i in range(len(test_labels)):
        all_distances = np.zeros(len(training_labels))
        for j in range(len(training_labels)):
            all_distances[j]=distance(test_input[:,i],training_input[:,j],p_final)
        ind = np.argpartition(all_distances,k_final)[:k_final]
        nearest_labels = [training_labels[temp] for temp in ind]
        pred_test_labels[i] = stats.mode(nearest_labels, axis=None)[0][0]
    num_correct = 0
    for ent in range(len(test_labels)):
        if (test_labels[ent]==pred_test_labels[ent]):
            num_correct += 1
    ACCURACY = num_correct/len(test_labels)
    TEST_PRED_LIST = pred_test_labels
    
    # clean up and return
    predictions = pd.DataFrame({'prediction': TEST_PRED_LIST}, dtype=np.int64)
    df = pd.concat([test, predictions], axis=1)
    return ACCURACY, df

if __name__ == '__main__':
    train = pd.read_csv('hw1data/propublicaTrain.csv')
    test = pd.read_csv('hw1data/propublicaTest.csv')
    
    print(perform_knn(train, test))