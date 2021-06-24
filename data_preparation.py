# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:24:42 2020

@author: mauro
"""

import pandas as pd
from pandas import DataFrame
from pandas import concat


def set_preparation_rop (surf_parameters):
    
    df_resampled = surf_parameters #imports cleaned and resampled well data
    
    df_rop = df_resampled[['ROP mph']]

    return df_rop

def set_preparation_swob (surf_parameters):
    
    df_resampled = surf_parameters #imports cleaned and resampled well data
    
    df_surface_test = df_resampled[['Bit Depth (MD) m', 'Weight on Bit kkgf', 'Average Hookload kkgf', 'Average Surface Torque kNm', 'Average Rotary Speed rpm', 'Average Standpipe Pressure kPa']]
 
    return df_surface_test


def set_preparation_calc_dwob (surf_parameters, calc_dwob):
    
    df_resampled = surf_parameters #imports cleaned and resampled well data
    df_DWOB = calc_dwob #imports calculated DWOB and Hookload values
    
    df_calc_downhole_test = df_resampled[['Bit Depth (MD) m']]
    df_calc_downhole_test['DWOB-Calc'] = df_DWOB['1']
    aux = df_resampled[['Average Hookload kkgf', 'Average Surface Torque kNm', 'Average Rotary Speed rpm', 'Average Standpipe Pressure kPa']]
    
    df_calc_downhole_test = pd.concat([df_calc_downhole_test, aux], axis=1)
    
    return df_calc_downhole_test


def step_preparation(reframed, n_load=1, n_prod=1, dropnan=True):
    variables = 1 if type(reframed) is list else reframed.shape[1]
    df = DataFrame(reframed)
    ax1, names = list(), list()
    # past times (t-n, ... t-1)
    for i in range(n_load, 0, -1):
        ax1.append(df.shift(i))
        names += [('param%d(t-%d)' % (j+1, i)) for j in range(variables)]
    # prediction (t, t+1, ... t+n)
    for i in range(0, n_prod):
        ax1.append(df.shift(-i))
        if i == 0:
            names += [('param%d(t)' % (j+1)) for j in range(variables)]
        else:
            names += [('param%d(t+%d)' % (j+1, i)) for j in range(variables)]
    # merge information
    total = concat(ax1, axis=1)
    total.columns = names
    # eliminate NaN values
    if dropnan:
        total.dropna(inplace=True)
    return total