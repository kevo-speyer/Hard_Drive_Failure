#!/bin/python

import numpy as np
import pandas as pd
import random

def get_relevant_att():
    """ Returns a list with the relevant attributes 
    T-test is performed with a significance of 0.05"""
    from scipy.stats import ttest_ind 
    #Get all data 
    all_data = pd.read_csv('harddrive.csv')
    # Get data from most used model
    ST400_data = all_data[all_data.model == 'ST4000DM000']
    neg_data = ST400_data[ST400_data.failure == 0 ]
    pos_data = ST400_data[ST400_data.failure == 1 ]
    n_pos = pos_data.shape[0]
    n_neg= neg_data.shape[0]

    rel_att = []
    for i_att in range(5,pos_data.shape[1]-5):
        if np.mod(i_att,2) == 0:
            continue
        p_val = ttest_ind(neg_data.iloc[:,i_att],pos_data.iloc[:,i_att],equal_var=False,nan_policy='omit')[1]
        if p_val < 0.05:
            rel_att.append(i_att)
            neg_nan_pctg = 100. * ( 1. - neg_data.iloc[:,i_att].count() / float(n_neg) )
            pos_nan_pctg = 100. * ( 1. - pos_data.iloc[:,i_att].count() / float(n_pos) )
            print "Relevant Attribute: ", i_att
            print "With p-value: ", p_val
            print "Nan % in healthy discs: " ,neg_nan_pctg
            print "Nan % in failure discs: " ,pos_nan_pctg
            print "Mean value for healthy HDD:",neg_data.iloc[:,i_att].mean(),"+/-", neg_data.iloc[:,i_att].std()
            print "Mean value for failure HDD:",pos_data.iloc[:,i_att].mean(),"+/-",pos_data.iloc[:,i_att].std()
            print ""

    return rel_att    

def get_corr_mat(ST400_data,rel_att):
    new_rel_att = []
    for index in rel_att:                       
        new_rel_att.append(index-3)

    corr_data = ST400_data.corr() 
    corr_mat = corr_data.iloc[new_rel_att,new_rel_att]

def load_data(load_fraction):
    """ return a random sample from total data. The number of records
    is = total_number_of_records * load_fraction """

    if load_fraction > 1:
        print "Error, load fraction must be <= 1"
        exit() 
    elif load_fraction < 0:
        print "Error, load fraction must be positive"
        exit()

    filename = "harddrive.csv"

    #number of records in file (excludes header)
    #n = sum(1 for line in open(filename)) - 1
    n = 3179295 
    s = int( load_fraction * n ) #desired sample size

    #the 0-indexed header will not be included in the skip list
    skip = sorted(random.sample(xrange(1,n+1),n-s)) 

    # DEBUG
    data = pd.read_csv(filename, skiprows=skip) # Use this line!
    #data = pd.read_csv("head_harddrive.csv")  # Using this line for HPC testing

    return data    

def get_most_common_models(data):
    mcm = (data.model.value_counts())[:] / float( data.model.shape[0])
    mcm = mcm[mcm>0.05]
    return mcm

def split_positive_data(data):
    #data[data.failure == 1].iloc[:,[0,1,2,3,4]]
    positive_data = data[data.failure == 1]
    negative_data = data[data.failure == 0]
    return positive_data, negaive_data


def select_column(data, column):
    array = data.iloc[:,[column]]
    return array

def get_col_stats(data):
    col_stats = data.describe
    for attrib in col_stats.columns:
        print col_stats[attrib]


def t_test(data):
    n_vals = df.groupby('failure').count() # count values for each attr 
    mean_vals = df.groupby('failure').mean() # get mean of each attr
    var_vals = df.groupby('failure').var() # get var of each attr
    neg_data = data[data.failure == 0]
    pos_data = data[data.failure == 1]
    ttest_ind(neg_data.iloc[:,12],pos_data.iloc[:,12])


rel_att = get_relevant_att()
print "Relavant attributes are (by index):", rel_att

    # Perform Welch test:
    #w_test = (mean_vals.iloc[0,:] - mean_vals.iloc[1,:])/np.sqrt( (var_vals/n_vals).iloc[0,:] + (var_vals/n_vals).iloc[1,:] )
    #
    ## Get Degrees Of Freedom for each attribute
    #dog = ( (var_vals.iloc[0,:] / n_vals.iloc[0,:] + var_vals.iloc[1,:] / n_vals.iloc[1,:])**2 ) / ( var_vals.iloc[0,:]**2 / ( n_vals.iloc[0,:]**2 * ( n_vals.iloc[0,:] - 1. ) ) + var_vals.iloc[1,:]**2 / ( n_vals.iloc[1,:]**2 * ( n_vals.iloc[1,:] - 1. ) ) )

##Load data
#load_pctg = 0.05
#data = load_data(load_pctg)
#pos_data, neg_data = split_positive_data(data)
#
## Get most common models
#mcm = get_most_common_models(data)
#mcm_pos = get_most_common_models(pos_data)
#print mcm
#print cmc_pos
#
##n_records = data.shape[0]
##print "n_records = ", n_records

# Panda Ticks
#data.iloc[:,[cols].describe #gives statistics of column
#data.groupby('failure').mean()
#



