# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:37:26 2020

@author: mauro
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import warnings
from numpy.random import seed
import tensorflow as tf

from data_normalization import well_norm, resampling
from hkld_dwob import hkld_corr, dwob_calc
from data_preparation import set_preparation_rop, set_preparation_swob, set_preparation_calc_dwob
from rnn_model_lstm1 import train_model


warnings.filterwarnings('ignore')
seed(0)
tf.random.set_seed(0)

df_test = pd.read_csv('Well_Data.csv') # Inputs: 'Bit Depth (MD) m', 'Weight on Bit kkgf', 'Average Hookload kkgf','Average Surface Torque kNm', 'MWD Downhole WOB', 'MWD Downhole Torque', 'Average Rotary Speed rpm', 'Mud Flow In lpm', 'Average Standpipe Pressure kPa', 'ROP mph'
BHA = pd.read_csv('BHA.csv').to_numpy()
Surveys = pd.read_csv('surveys.csv').to_numpy()
dwob = pd.read_csv('Calc_DWOB2.csv') #Calcualted DWOB

memory = 20
predictions = 20
percent_drilled = np.arange(15,81,1)

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
#X_prep = set_preparation_swob(surf_parameters_i) #Prepare data set with Features.
X1_prep = set_preparation_rop(surf_parameters_i) #Prepare data set with only ROP information

histories = []
val_loss_array = []
test_loss_array = []
y_test_array= []
y_pred_array = []
aae_array = []

for i in percent_drilled:
    
    print (f'Evaluating {i-14}/{len(percent_drilled)}')
    
    percent = i/100 
    
    data_length = len(X_prep)-memory-predictions # Necessary to avoid repeating information when data is shifted 
    
    val_loss, test_loss, aae, y_test, y_pred = train_model(percent, data_length, memory, predictions, X_prep, X1_prep)
        
    val_loss_array.append(val_loss)
    test_loss_array.append(test_loss)
    aae_array.append(aae)
    y_test_array.append(y_test)
    y_pred_array.append(y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

np.save('aae_array_16_06_DWOB_27', aae_array)
np.save('test_array_16_06_DWOB_27', y_test_array)
np.save('pred_array_16_06_DWOB_27', y_pred_array)

sns.heatmap(aae_array)