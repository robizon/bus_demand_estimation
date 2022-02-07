# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:41:06 2021

@author: robizon
"""

import random
import math
import pandas as pd
import numpy as np

import statsmodels.api as sm

import statsmodels.discrete.count_model as cm

from patsy import dmatrices

import statsmodels.formula.api as smf


def poisson_model(data):
    
    
        
    data_model = data[['stop_5', 'pre3stop', 'rush_hour']]
       

    
      
    model=smf.glm(formula = "stop_5 ~ pre3stop + rush_hour", data=data_model, family=sm.families.Poisson()).fit()
    
    return model  



RMSE=[]

#[1.25, 2.5, 5, 10, 20, 40, 80]
for r_h_freq in [1.5625, 3.125, 6.25, 12.5, 25, 50, 100, 200]:
    
    l_5=5
    
    RMSE_temp=[]
    
    
    RMSE_temp.append(round(1/r_h_freq, 4)*100)
    
    random.seed(10)
    
    
    
    lambdas = [3, 4, 5, 4, l_5]
    
    bus_capacity=sum(lambdas)+2*sum(lambdas)/len(lambdas)
    
    _num_intervals = 1000000
    
    arrivals=pd.DataFrame()
    
    rush_our_frequency=int(_num_intervals/r_h_freq)
    
    
    for _stop_number in range(len(lambdas)): 
        s = np.random.poisson(lambdas[_stop_number], _num_intervals-rush_our_frequency)
        s=pd.Series(s)
        s=s.rename('stop_'+str(_stop_number+1))
        arrivals=pd.concat([arrivals, s], axis=1)
    
    
    arrivals_rush=pd.DataFrame()
    
    for _stop_number in range(len(lambdas)): 
        s = np.random.poisson(2*lambdas[_stop_number], rush_our_frequency)
        s=pd.Series(s)
        s=s.rename('stop_'+str(_stop_number+1))
        arrivals_rush=pd.concat([arrivals_rush, s], axis=1)
    
    
    arrivals=pd.concat([arrivals, arrivals_rush])
    
    
    arrivals=arrivals.reset_index(drop=True)
    
    arrivals_true=arrivals.copy()
    
    
    arrivals=arrivals.cumsum(axis=1)
    bus_full=arrivals>bus_capacity
    
    last_stop_full=bus_full['stop_5']
    
    arrivals=bus_capacity-arrivals
    arrivals=arrivals_true+arrivals
    arrivals=pd.concat([arrivals_true, arrivals]).min(level=0)
    arrivals=round(arrivals)
    
    arrivals[arrivals<0]=0
    
    print('True means')
    print(arrivals_true.mean())
    
    print('Data means')
    print(arrivals.mean())
    
    
    
    
    arrivals['rush_hour']=0
    arrivals.loc[_num_intervals-rush_our_frequency:, 'rush_hour']=1
    
    arrivals_true['rush_hour']=0
    arrivals_true.loc[_num_intervals-rush_our_frequency:, 'rush_hour']=1
    
    arrivals['bus_full']=0
    arrivals.loc[last_stop_full==1, 'bus_full']=1
    
    
    arrivals['pre3stop']=arrivals['stop_2']+arrivals['stop_3']+arrivals['stop_4']
    arrivals_true['pre3stop']=arrivals_true['stop_2']+arrivals_true['stop_3']+arrivals_true['stop_4']
    
    msk = np.random.rand(len(arrivals)) < 0.8
    
    train = arrivals.iloc[msk]
    test = arrivals_true.iloc[~msk]
    
    
    '''
    Poisson no exlusions
    '''
    
    poisson = poisson_model(train)
    print(poisson.summary())
    poisson_predictions = poisson.predict(test)
    #print(poisson_predictions.summary_frame())
    print('Poisson no exlusions')
    rmse=sum((test['stop_5']-poisson_predictions)**2)/len(test['stop_5'])
    rmse=rmse**0.5
    print('RMSE '+str(rmse))
    
    RMSE_temp.append(rmse)
    
    
    '''
    Poisson with excluding zeros when bus is full
    '''
    
    
    poisson = poisson_model(train[~((train['bus_full']==1) & (train['stop_5']==0))])
    print(poisson.summary())
    poisson_predictions = poisson.predict(test)
    #print(poisson_predictions.summary_frame())
    print('Poisson no cencored zeros')
    rmse=sum((test['stop_5']-poisson_predictions)**2)/len(test['stop_5'])
    rmse=rmse**0.5
    print('RMSE '+str(rmse))
    
    RMSE_temp.append(rmse)
    
    
    '''
    Poisson with excluding cencored data when bus is full
    '''
    
    
    poisson = poisson_model(train[train['bus_full']==0])
    print(poisson.summary())
    poisson_predictions = poisson.predict(test)
    #print(poisson_predictions.summary_frame())
    print('Poisson no cencoring')
    rmse=sum((test['stop_5']-poisson_predictions)**2)/len(test['stop_5'])
    rmse=rmse**0.5
    print('RMSE '+str(rmse))
    
    RMSE_temp.append(rmse)
    
    
    

    
    RMSE.append(RMSE_temp)
    
    
    
    
    


RMSE=pd.DataFrame(RMSE, columns=['rush_hour_frequency', 'poisson', 'no_cencored_zeros', 'no_cencoring'])


#RMSE.to_csv('RMSE_cap_sum_plus_2mean_new.csv')
