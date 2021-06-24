# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:29:56 2020

@author: Mauro Encinas

"""

import pandas as pd
import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor

# In[1]: Data Normalization
    
def well_norm(well_data):

# Importing Data
    
    df_test = well_data
    df_test=df_test.rename(columns = {'0':'Bit Depth (MD) m','1':'Weight on Bit kkgf','2':'Average Hookload kkgf','3':'Average Surface Torque kNm','4':'MWD Downhole WOB','5':'MWD Downhole Torque','6':'Average Rotary Speed rpm','7':'Mud Flow In lpm','8':'Average Standpipe Pressure kPa','9':'ROP mph'})
    

# IQR Outlier Removal

    Q1_DWOB = df_test['MWD Downhole WOB'].quantile(0.25)
    Q3_DWOB = df_test['MWD Downhole WOB'].quantile(0.95)
    IQR_DWOB = Q3_DWOB - Q1_DWOB
    Q1_SWOB = df_test['Weight on Bit kkgf'].quantile(0.25)
    Q3_SWOB = df_test['Weight on Bit kkgf'].quantile(0.95)
    IQR_SWOB = Q3_SWOB - Q1_SWOB
    Q1_HKLD = df_test['Average Hookload kkgf'].quantile(0.25)
    Q3_HKLD = df_test['Average Hookload kkgf'].quantile(0.90)
    IQR_HKLD = Q3_HKLD - Q1_HKLD
    Q1_STQ = df_test['Average Surface Torque kNm'].quantile(0.25)
    Q3_STQ = df_test['Average Surface Torque kNm'].quantile(0.95)
    IQR_STQ = Q3_STQ - Q1_STQ
    Q1_DTQ = df_test['MWD Downhole Torque'].quantile(0.25)
    Q3_DTQ = df_test['MWD Downhole Torque'].quantile(0.95)
    IQR_DTQ = Q3_DTQ - Q1_DTQ
    Q1_ROP = df_test['ROP mph'].quantile(0.25)
    Q3_ROP = df_test['ROP mph'].quantile(0.95)
    IQR_ROP = Q3_ROP - Q1_ROP
    Q1_RPM = df_test['Average Rotary Speed rpm'].quantile(0.25)
    Q3_RPM = df_test['Average Rotary Speed rpm'].quantile(0.95)
    IQR_RPM = Q3_RPM - Q1_RPM
    Q1_LXM = df_test['Mud Flow In lpm'].quantile(0.25)
    Q3_LXM = df_test['Mud Flow In lpm'].quantile(0.95)
    IQR_LXM = Q3_LXM - Q1_LXM
    Q1_PPR = df_test['Average Standpipe Pressure kPa'].quantile(0.25)
    Q3_PPR = df_test['Average Standpipe Pressure kPa'].quantile(0.95)
    IQR_PPR = Q3_PPR - Q1_PPR
    
    df_test = df_test.query('(@Q1_DWOB - 1.5 * @IQR_DWOB) <= `MWD Downhole WOB` <= (@Q3_DWOB + 1.5 * @IQR_DWOB)')
    df_test = df_test.query('(@Q1_SWOB - 1.5 * @IQR_SWOB) <= `Weight on Bit kkgf` <= (@Q3_SWOB + 1.5 * @IQR_SWOB)')
    df_test = df_test.query('(@Q1_HKLD - 1.5 * @IQR_HKLD) <= `Average Hookload kkgf` <= (@Q3_HKLD + 1.5 * @IQR_HKLD)')
    df_test = df_test.query('(@Q1_STQ - 1.5 * @IQR_STQ) <= `Average Surface Torque kNm` <= (@Q3_STQ + 1.5 * @IQR_STQ)')
    df_test = df_test.query('(@Q1_DTQ - 1.5 * @IQR_DTQ) <= `MWD Downhole Torque` <= (@Q3_DTQ + 1.5 * @IQR_DTQ)')
    df_test = df_test.query('(@Q1_ROP - 1.5 * @IQR_ROP) <= `ROP mph` <= (@Q3_ROP + 1.5 * @IQR_ROP)')
    df_test = df_test.query('(@Q1_RPM - 1.5 * @IQR_RPM) <= `Average Rotary Speed rpm` <= (@Q3_RPM + 1.5 * @IQR_RPM)')
    df_test = df_test.query('(@Q1_LXM - 1.5 * @IQR_LXM) <= `Mud Flow In lpm` <= (@Q3_LXM + 1.5 * @IQR_LXM)')
    df_test = df_test.query('(@Q1_PPR - 1.5 * @IQR_PPR) <= `Average Standpipe Pressure kPa` <= (@Q3_PPR + 1.5 * @IQR_PPR)')
    

# Moving Average Filter

    df_test['Bit Depth (MD) m'] = df_test['Bit Depth (MD) m'].rolling(window=30,center=True).mean()
    df_test['Weight on Bit kkgf'] = df_test['Weight on Bit kkgf'].rolling(window=30,center=True).mean()
    df_test['Average Hookload kkgf'] = df_test['Average Hookload kkgf'].rolling(window=30,center=True).mean()
    df_test['Average Surface Torque kNm'] = df_test['Average Surface Torque kNm'].rolling(window=30,center=True).mean()
    df_test['MWD Downhole WOB'] = df_test['MWD Downhole WOB'].rolling(window=30,center=True).mean()
    df_test['MWD Downhole Torque'] = df_test['MWD Downhole Torque'].rolling(window=30,center=True).mean()
    df_test['Average Rotary Speed rpm'] = df_test['Average Rotary Speed rpm'].rolling(window=30,center=True).mean()
    df_test['Mud Flow In lpm'] = df_test['Mud Flow In lpm'].rolling(window=30,center=True).mean()
    df_test['Average Standpipe Pressure kPa'] = df_test['Average Standpipe Pressure kPa'].rolling(window=30,center=True).mean()
    df_test['ROP mph'] = df_test['ROP mph'].rolling(window=30,center=True).mean()

    df_test.dropna(inplace = True)
    df_test.reset_index(drop=True,inplace=True)
    
    return df_test

# In[2]: Data Resampling
    
def resampling (df_test,leaf_size,radius):
    
    RN = df_test.values
    X = RN[0:,0].reshape(-1, 1)
    y = RN[0:,1:10].reshape(-9, 9)
    
    neigh = RadiusNeighborsRegressor(leaf_size=leaf_size, radius=radius, weights='uniform')
    neigh.fit(X, y)
    
    X_test = np.arange(1900.677364, 2399.972340, 0.249647488).reshape(-1, 1) # was 0.099858995 0.249647488
    X_test.shape
    
    param = neigh.predict(X_test)
    
    Result = np.concatenate((X_test,param),axis=1)
    Resampled = pd.DataFrame(Result)
    Resampled = Resampled.rename(columns = {0:'Bit Depth (MD) m',1:'Weight on Bit kkgf',2:'Average Hookload kkgf',3:'Average Surface Torque kNm',4:'MWD Downhole WOB',5:'MWD Downhole Torque',6:'Average Rotary Speed rpm',7:'Mud Flow In lpm',8:'Average Standpipe Pressure kPa',9:'ROP mph'})
    Resampled.dropna(inplace=True)
    Resampled.reset_index(drop=True,inplace=True)
    
    return Resampled