#!/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def train_logistic_regresion(neg_data,pos_data,rel_att_2):
    """ input data, output trained model """
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    rel_att_2=[9, 15, 19, 37, 41, 49] # keep relevant attributes

    pos_data_array = pos_data.iloc[:,rel_att_2].values   
    neg_data_array = neg_data.iloc[:,rel_att_2].values

    X_data = np.zeros([pos_data_array.shape[0]+neg_data_array.shape[0],len(rel_att_2)])
    X_data[0:neg_data_array.shape[0],:] = neg_data_array
    X_data[neg_data_array.shape[0]:,:] = pos_data_array

    Y_data = np.zeros(pos_data_array.shape[0]+neg_data_array.shape[0])   
    Y_data[0:neg_data_array.shape[0]] = 0
    Y_data[neg_data_array.shape[0]:] = 1

    #Rescale attributes between  -1 and 1
    scaler = StandardScaler()    
    X_std = scaler.fit_transform(X_data) 

    # choose class_weight='balanced', to give more weight to positive data
    clf = LogisticRegression(random_state=0, class_weight='balanced')
    
    # Train model
    model = clf.fit(X_std, Y_data)

    return model

def plt_ROC_curve(model, X_std, Y_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
   
    fp_ls = []   
    fn_ls = []
   
    for thrshld in np.linspace(0.0, 1.0, num=201):
        Y_pred = ((model.predict_proba(X_std))[:,1]+thrshld).astype(int)
        con_mat = confusion_matrix(Y_data, Y_pred)
        falso_negativo = con_mat[1,0]/float(sum(con_mat[1,:]))
        falso_positivo = con_mat[0,1]/float(sum(con_mat[0,:]))
        fn_ls.append(falso_negativo)
        fp_ls.append(falso_positivo)


    plt.plot(np.array(fp_ls),1.-np.array(fn_ls),color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=12)
    plt.xlabel('False Positive / (Flase Pos + True Negative)')
    plt.ylabel('True Positive / (False Negative + True Positive)')
    plt.show()

def read_discr_data():
    """ Read data, keep just 1 model and return 2 sets discriminated by label"""
    
    #Get all data 
    all_data = pd.read_csv('harddrive.csv')

    # Get most used model
    most_used_model = ((all_data.model.value_counts() / all_data.shape[0])[0:1]).index[0]
    # Get data only for most used model
    ST400_data = all_data[ all_data.model == most_used_model ]

    neg_data = ST400_data[ST400_data.failure == 0 ]
    pos_data = ST400_data[ST400_data.failure == 1 ]

    return pos_data, neg_data

def get_relevant_att(pos_data, neg_data):
    """ Returns a list with the relevant attributes 
    T-test is performed with a significance of 0.05"""
    from scipy.stats import ttest_ind 

    n_pos = pos_data.shape[0]
    n_neg = neg_data.shape[0]

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


#def get_rel_att_2(pos_data, neg_data):
#    """The idea is to extract the attributes by comparing the probability distributions
#    in each group"""
#    chi2_of_each_att = []
#    for i_att in atributes:
#        pos_pdf = get_pdf(pos_data.iloc[:,i_att])
#        neg_pdf = get_pdf(neg_data.iloc[:,i_att])
#        align indexes of pos_pdf and neg_pdf
#        sum_dummy = 0.
#        for i_value in all_values_of_i_att:
#            sum_dummy += ( pos_pdf[i_value] - neg_pdf[i_value] ) ** 2
#
#        chi2_of_each_att[i_att] = sum_dummy
#    
#    order attrbiutes by value of chi2_of_each_att
#    return oreder list of most relevant attributes

def get_corr_mat(ST400_data,rel_att):
    new_rel_att = []
    for index in rel_att:                       
        new_rel_att.append(index-3)

    corr_data = ST400_data.corr() 
    corr_mat = corr_data.iloc[new_rel_att,new_rel_att]

    return corr_mat

def plot_distros(rel_att,pos_data,neg_data):
    """Plot the histograms of each of the relative attributes, discriminated by label of failure"""
    for i_att in rel_att:
        plt.plot(neg_data.iloc[:,i_att].value_counts()/neg_data.shape[0],'ro',pos_data.iloc[:,i_att].value_counts()/pos_data.shape[0],'gP')
        plt.ylabel('probability')
        plt.xlabel(neg_data.columns[i_att])
        plt.show()    

    return

def plot_2var(rel_att,pos_data,neg_data):
    """make 2D Plot of relevant variable vs relevant varialbe with colored filures"""
    for i_att in range(len(rel_att)-1):
        for j_att in range(i_att+1,len(rel_att)):
            plt.plot(neg_data.iloc[:,rel_att[i_att]],neg_data.iloc[:,rel_att[j_att]],'ro',pos_data.iloc[:,rel_att[i_att]],pos_data.iloc[:,rel_att[j_att]],'gP')
            plt.ylabel(neg_data.columns[ rel_att[j_att] ])
            plt.xlabel(neg_data.columns[ rel_att[i_att] ])
            plt.show()

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


#rel_att = get_relevant_att()
#print "Relavant attributes are (by index):", rel_att

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



