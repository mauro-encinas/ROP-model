# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:05:11 2020

@author: atunkiel
"""
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
from numpy.random import seed
import tensorflow as tf

from data_normalization import well_norm, resampling
from hkld_dwob import hkld_corr, dwob_calc
from data_preparation import set_preparation_rop, set_preparation_calc_dwob, step_preparation
#from rnn_model import train_model


warnings.filterwarnings('ignore')
seed(0)
tf.random.set_seed(0)

df_test = pd.read_csv('Well_Data.csv') # Inputs: 'Bit Depth (MD) m', 'Weight on Bit kkgf', 'Average Hookload kkgf','Average Surface Torque kNm', 'MWD Downhole WOB', 'MWD Downhole Torque', 'Average Rotary Speed rpm', 'Mud Flow In lpm', 'Average Standpipe Pressure kPa', 'ROP mph'
BHA = pd.read_csv('BHA.csv').to_numpy()
Surveys = pd.read_csv('surveys.csv').to_numpy()
dwob = pd.read_csv('Calc_DWOB.csv') #Calcualted DWOB

memory = 100
predictions = 100
percent_drilled = np.arange(79,81,1)

## Pre-process complete data set

normalized = well_norm(df_test)

resampled = resampling(normalized, 100, 0.3)

hkld = hkld_corr(resampled)

#dwob = dwob_calc(hkld, BHA, Surveys)

surf_parameters_i = resampled
corr_hkld_i = hkld
calc_dwob_i = dwob

## Set preparation of Features and Objective

X_prep = set_preparation_calc_dwob(surf_parameters_i,calc_dwob_i) #Prepare data set with Features.
X1_prep = set_preparation_rop(surf_parameters_i) #Prepare data set with only ROP information

histories = []
val_loss_array = []
test_loss_array = []
aae_array = []
#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

import pandas as pd

percentage_drilled = np.arange(15,81,1)

results_aae = []

print('#' * len(percentage_drilled))
for i in percentage_drilled:
    print('.', end='')
    #reg = GradientBoostingRegressor()
    #reg = RandomForestRegressor(n_estimators = 70, random_state = 0,max_depth=30)
    #reg = KNeighborsRegressor(algorithm='brute', leaf_size=30, metric='minkowski', n_neighbors=3, weights='distance')
    reg  = XGBRegressor()
    
    border1 = int(len(X1_prep)*i/100)
    border2 = int(len(X1_prep)*(i+20)/100)
    
    reg.fit(X_prep[:border1], X1_prep[:border1])
    y_pred = reg.predict(X_prep[border1:border2])
    y_true = X1_prep[border1:border2].to_numpy().ravel()
    results_aae.append(np.abs(y_pred - y_true))
    
    np.save('basic_XBR', results_aae)

#%%


average_at_percentage = np.average(results_aae, axis=1)

plt.plot(average_at_percentage)

#%%
# figure SWOB 5m prediction
fig, ax = plt.subplots(figsize=(9,7))
# plot heatmap
sns.heatmap(results_aae, cmap='RdYlGn_r',vmax=8)
#ax.set_xticks(ax.get_xticks()[::2])
#ax.set_yticks(ax.get_yticks()[::8])
#ax.set_xlabel('Predicted Length (m)')
#ax.set_ylabel('Depth (m)')