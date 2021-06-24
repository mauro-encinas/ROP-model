# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:31:13 2020

@author: mauro
"""

import pandas as pd
import numpy as np
from math import sin, cos

def hkld_corr (resampled):
    
    df_resampled = resampled
    
# Sheave Effect - Inactive Dead-Line Sheave

    df_hook = df_resampled[['Bit Depth (MD) m', 'Average Hookload kkgf','Weight on Bit kkgf','MWD Downhole WOB','Average Standpipe Pressure kPa']]
    df_hook.dropna(inplace = True)
    df_hook.reset_index(drop=True,inplace=True)
    
    def Wcorr(Fdl,e):
        n=16 #number of lines
        return (Fdl/n)*(1-e**n)/(1-e)



    # Sensitivity Analysis Sheave Efficiency according to literature vary from 96-99%
    W_corr_1 = Wcorr(df_hook[['Average Hookload kkgf']],0.96).rename(columns = {'Average Hookload kkgf':'HKLD_A1 96% kkgf'})
    W_corr_2 = Wcorr(df_hook[['Average Hookload kkgf']],0.97).rename(columns = {'Average Hookload kkgf':'HKLD_A1 97% kkgf'})
    W_corr_3 = Wcorr(df_hook[['Average Hookload kkgf']],0.98).rename(columns = {'Average Hookload kkgf':'HKLD_A1 98% kkgf'})
    W_corr_4 = Wcorr(df_hook[['Average Hookload kkgf']],0.99).rename(columns = {'Average Hookload kkgf':'HKLD_A1 99% kkgf'})

    HL_a_1 = pd.concat([W_corr_1, W_corr_2, W_corr_3, W_corr_4],axis=1)

# Static Hook Effect
    
    df_slips = pd.read_csv('Block_weight_memory.csv') # Hook load recorded when "In Slips" in whole well
    df_slips.describe()
    
    def BLKW(hkld, e):
        n=16
        return (hkld/n)*(1-e**n)/(1-e)
    
    # Sensitivity Analysis Sheave Efficiency according to literature vary from 96-99%

    BLKW_corr_1 = BLKW(df_slips[['Traveling Block Weight kkgf']].min(),0.96)
    BLKW_corr_2 = BLKW(df_slips[['Traveling Block Weight kkgf']].min(),0.97)
    BLKW_corr_3 = BLKW(df_slips[['Traveling Block Weight kkgf']].min(),0.98)
    BLKW_corr_4 = BLKW(df_slips[['Traveling Block Weight kkgf']].min(),0.99)
    
    HL_a = pd.concat([BLKW_corr_1, BLKW_corr_2, BLKW_corr_3, BLKW_corr_4],axis=1)
    HL_a_2 = pd.DataFrame(np.repeat(HL_a.values,len(df_hook['Average Hookload kkgf']),axis=0))
    HL_a_2.columns = ['HKLD_A2 96% kkgf', 'HKLD_A2 97% kkgf', 'HKLD_A2 98% kkgf','HKLD_A2 99% kkgf']

# Stand Pipe Pressure Effect

    df_pump = df_hook[['Bit Depth (MD) m', 'Average Standpipe Pressure kPa']] / 6.89475 # Turn KPa to psi

    def Press(P1):
        Id=4.67
        return (5.095*10**(-5))*P1*(Id**2)*1.019716 ## To convert from KdaN to kkgf


    HL_a_3 = Press(df_pump[['Average Standpipe Pressure kPa']])
    HL_a_3=HL_a_3.rename(columns = {'Average Standpipe Pressure kPa':'HKLD_A3 kkgf'})

# Calculate the Real Hookload

    Real_HKLD = df_hook[['Bit Depth (MD) m']]
    Real_HKLD['SWOB kips'] = (df_hook[['Weight on Bit kkgf']])*2.2046 #conversion from tons to kip
    Real_HKLD['DWOB kips'] = (df_hook[['MWD Downhole WOB']])*2.2046 #conversion from tons to kip
    Real_HKLD['HKLD-96%'] = (HL_a_1['HKLD_A1 96% kkgf'] - HL_a_2['HKLD_A2 96% kkgf'] - HL_a_3['HKLD_A3 kkgf'])*2.2046
    Real_HKLD['HKLD-97%'] = (HL_a_1['HKLD_A1 97% kkgf'] - HL_a_2['HKLD_A2 97% kkgf'] - HL_a_3['HKLD_A3 kkgf'])*2.2046
    Real_HKLD['HKLD-98%'] = (HL_a_1['HKLD_A1 98% kkgf'] - HL_a_2['HKLD_A2 98% kkgf'] - HL_a_3['HKLD_A3 kkgf'])*2.2046
    Real_HKLD['HKLD-99%'] = (HL_a_1['HKLD_A1 99% kkgf'] - HL_a_2['HKLD_A2 99% kkgf'] - HL_a_3['HKLD_A3 kkgf'])*2.2046
    
    
    Real_HKLD.to_csv('Corrected_hookload.csv',index=False)
    
    return Real_HKLD

def dwob_calc (Real_HKLD, BHA, Surveys):
    
    BHA = BHA
    # BHA components from Bottom to Top (Bit-DP). Units: Length (m), OD(m), ID(m), Weight (N/m).
    Surveys = Surveys
    # Trajectory of the well. Depth(m), Inc.(rad), Az.(rad).
    Buoyancy = 0.81 #Assuming equal density inside and outside (1-Df/Ds)
    Friction_factor = np.array([0.22,0.18]) # Open Hole and Casing-Riser [dimensionless]
    WOB = 0
    
# Hookload function

    def real_hookload(Surveys, BHA, Buoyancy, WOB):

        T_D = np.zeros((len(Surveys),12))
    
        for i in reversed(range(1,len(T_D))):
            T_D[i,0] = Surveys[i,0] #Depth(m)
            T_D[i,1] = Surveys[i,1] #Inclination[rad]
            T_D[i,2]= Surveys[i,2] #Azimuth[rad]
            T_D[i-1,3] = (Surveys[i,1] + Surveys[i-1,1])/2 #Average Inclination ((I1+I2)/2) [rad]
            T_D[i-1,4] = Surveys[i,1] - Surveys[i-1,1] #Inclination Diff (I2-I1) [rad]
            T_D[i-1,5]= Surveys[i,2] - Surveys[i-1,2] #Azimuth Diff (A2-A1) [rad]
            T_D[i-1,6]= Surveys[i,0] - Surveys[i-1,0] #Depth Difference (D2-D1) [m]
            if Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]): #Determine the Weight of the String in each cell [N/m]
                T_D[i-1,7]= BHA[0,3] #DD
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]):
                T_D[i-1,7]= BHA[1,3] #LWB
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]):
                T_D[i-1,7]= BHA[2,3] #NMDC
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]):
                T_D[i-1,7]= BHA[3,3] #STB
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]-BHA[4,0]):
                T_D[i-1,7]= BHA[4,3] #DC
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]-BHA[4,0]-BHA[5,0]):
                T_D[i-1,7]= BHA[5,3] #DC2
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]-BHA[4,0]-BHA[5,0]-BHA[6,0]):
                T_D[i-1,7]= BHA[6,3] #Jar
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]-BHA[4,0]-BHA[5,0]-BHA[6,0]-BHA[7,0]):
                T_D[i-1,7]= BHA[7,3] #DC3
            elif Surveys[i,0] > (Surveys[-1,0]-BHA[0,0]-BHA[1,0]-BHA[2,0]-BHA[3,0]-BHA[4,0]-BHA[5,0]-BHA[6,0]-BHA[7,0]-BHA[8,0]):
                T_D[i-1,7]= BHA[8,3] #HWDP
            else:
                T_D[i-1,7]= BHA[9,3] #DP
            if Surveys[i,0] > 1405: 
                T_D[i-1,8]= Friction_factor[0] #OpenHole
            else: 
                T_D[i-1,8]= Friction_factor[1] #CasedHole
    
        Incremental_down = -WOB #Set lower boundary condition [N]
        Incremental_up = -WOB #Set lower boundary condition [N]
        Rotating = -WOB #Set lower boundary condition [N]

        for i in reversed(range(len(T_D))):
            T_D[i,9] = Incremental_down #Incremental force downwards [N]
            Incremental_down += Buoyancy * T_D[i-1,7] * T_D[i-1,6] * cos(T_D[i-1,3]) - T_D[i-1,8]*(((Incremental_down*T_D[i-1,5]*sin(T_D[i-1,3]))**2+(Incremental_down*T_D[i-1,4]+Buoyancy * T_D[i-1,7] * T_D[i-1,6]*sin(T_D[i-1,3]))**2)**.5)

            T_D[i,10] = Incremental_up #Incremental force upwards [N]
            Incremental_up += Buoyancy * T_D[i-1,7] * T_D[i-1,6] * cos(T_D[i-1,3]) + T_D[i-1,8]*(((Incremental_up*T_D[i-1,5]*sin(T_D[i-1,3]))**2+(Incremental_up*T_D[i-1,4]+Buoyancy * T_D[i-1,7] * T_D[i-1,6]*sin(T_D[i-1,3]))**2)**.5)
    
            T_D[i,11] = Rotating #Hookload when Rotating [N]
            Rotating += Buoyancy * T_D[i-1,7] * T_D[i-1,6] * cos(T_D[i-1,3])
    
        return Rotating/4448.22162
    
# Function that Create Survey Tables for a Certain Bit Depth

    def int_svy(Y, Z):

        A = np.zeros((len(Y),3)) #creating a table with the dimension of the surveys
    
        for i in reversed(range(len(Y))):
        
            if  Y[i-1,0] <= Z <= Y[i,0]: # finds the position of the bit between the surveys
                x = Y[i,0] - Z
                y = Z - Y[i-1,0]
                if x < y: #choose the closest value (upper-lower) and equates the inclination and azimuth [rad]
                    A[i,0] = Z
                    A[i,1] = Y[i,1]
                    A[i,2] = Y[i,2]
                else:
                    A[i,0] = Z
                    A[i,1] = Y[i-1,1]
                    A[i,2] = Y[i-1,2]
            elif Z <= Y[i,0]: #if the survey depth is bigger than the bit position, the row is eliminated
                A = np.delete(A, i, axis=0)
            else: #if the bit depth is bigger than the other surveys depth, copy the survey depth, iclination and azimuth
                A[i,0] = Y[i,0]
                A[i,1] = Y[i,1]
                A[i,2] = Y[i,2]
        return A
    
    Corrected = Real_HKLD #Import Corrected Hookload Table

    N = Corrected.iloc[0:,:3].to_numpy() # Select: 'Bit Depth (MD) m', 'SWOB kips', 'DWOB kips'

    C = Corrected.iloc[0:,0::6].to_numpy() # Select: 'Bit Depth (MD) m', 'HKLD-99%'
    
# "Shooting Method" function that calculates the DWOB for each data point
    
    D = np.zeros((len(N),3),dtype=np.float64) #Reserve space for Bit depth and Rotating Results

    for i in reversed(range(len(D))): #iterates within the table in reverse order    
        D[i,0] = N[i,0] #Set the value of the bit depth  
        D[i,1] = N[i,1] #Initially set the value of the WOB with the SWOB, wich will be changed, if we don't meet the evaluation criteria
        Traject = int_svy(Surveys,D[i,0]) #Run the trajectory function for each bit depth  
        Rot = real_hookload(Traject, BHA, Buoyancy, D[i,1]) #Run the T&D function for each bit depth, returns a Hooklad 
        D[i,2] = Rot #Stores the value of the Hookload for the condition stated above [kips]\n
        while abs(D[i,2]-C[i,1])> .5 : #While the difference between the calculated and corrected value is bigger than .5 [kips]
            if (D[i,2]-C[i,1]) > 0: #Calculated value by T&D is bigger than the corrected value\n
                D[i,1] += 4448.22162/4 # Increase 0.25 [kips] WOB\n
            else:
                D[i,1] -= 4448.22162/4 # Decrease 0.25 [kips] WOB\n
            Rot = real_hookload(Traject, BHA, Buoyancy, D[i,1])
            D[i,2] = Rot
    
    
    D [0:,1] = D [0:,1]*0.000101971621 # Convert WOB from [N] to [Tonnes]
    D [0:,2] = D[0:,2]*0.45359237 # Convert Calculated Hook Load from [kips] to [Tonnes]
    
    DWOB_calc = pd.DataFrame(D)
    
    return DWOB_calc
