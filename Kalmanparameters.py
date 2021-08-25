# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:21:48 2021

@author: panay
"""
import pandas as pd
import numpy as np
from numpy.random import randint 
import plotly.graph_objects as go

from Kalman_2D_scratch import KalmanFilter

def noise(x0,a):
    mu = a/2 # [0,q) mean
    std_noise= a-mu # q-mu 
    
    return np.random.normal(mu,std_noise,size=(len(x0),1))#uses abs to get [0,

def Ynoise(df_y,e):
    
    mu = e/2 # [0,s) mean
    std_noise= e-mu # s-mu
    
    return np.random.normal(mu,std_noise,size=(len(df_y.loc[0]),1))[0] # y is a single value so getting a single value    

df=pd.read_csv(r'./Simulated_Values.csv')
#have the Y values here
df_x=df.drop('Y_random',axis=1)
df_x=df.drop('Y',axis=1)
df_y=df[['Y']]
#Initialise Kalman filter
Kf= KalmanFilter()

#Initial guess uses the mean of the df
x0 = df_x.mean().values.reshape(len(df_x.loc[0]),1)


#Process noise
q=5

#The errors are set only once as assumed they do not change each iteration
#uses the standard deviation for initial error
Estimate_error= df_x.std().values.reshape(len(df_x.loc[0]),1)
# Estimate Uncertainty
p0 = pow(Estimate_error,2)


#measurement noise
r=2
  
 

True_Y=[]
Ys=[]
Ys_Variances=[]

Predictions=[]
Pred_Variances=[]

Predictions2=[]
Pred_Variances2=[]

Kalmangain=[]

Measurements=[]
Measurements2=[]

Current_State_Estimate=[]
Current_State_Variance=[]

Current_State_Estimate2=[]
Current_State_Variance2=[]

#Kalman filter parameters
Parameter={
    'x0':df_x.mean().values.reshape(len(df_x.loc[0]),1),# Initial state guess
    'F': np.eye(len(x0)),# State transition matrix
    'G': np.zeros(len(x0)),# Control Matrix
    'u': np.zeros(len(x0)),# Input variable
    'Q': np.eye(len(x0)) * noise(x0,q),# Process uncertainty 
    'P': np.eye(len(p0))*p0,# Initial Covariance guess
    'R': np.eye(len(x0))* noise(x0,r),# Measurment uncertainty 
    'H': np.eye(len(df_x.loc[0])),#Observation matrix
    
    }
# identity matrix
I = np.eye(len(Parameter.get('H')))


# '''
# #How the values were simulated
# x1=randint(100, size=100)
# x2=randint(100, size=100)
# y=randint(100, size =100)

# y is not random anymore it is the summation of x1 and x2
# ''' 

def get_Parameters(df_x,df_y,q,r):
    #Initial guess uses the mean of the df
    x0 = df_x.mean().values.reshape(len(df_x.loc[0]),1)
    
    
    #Process noise
    q=5
    
    #The errors are set only once as assumed they do not change each iteration
    #uses the standard deviation for initial error
    Estimate_error= df_x.std().values.reshape(len(df_x.loc[0]),1)
    # Estimate Uncertainty
    p0 = pow(Estimate_error,2)
    
    
    #measurement noise
    r=2
      
    
    #Kalman filter parameters
    Parameter={
        'x0':df_x.mean().values.reshape(len(df_x.loc[0]),1),# Initial state guess
        'F': np.eye(len(x0)),# State transition matyrix
        'G': np.zeros(len(x0)),# Control Matrix
        'u': np.zeros(len(x0)),# Input variable
        'Q': np.eye(len(x0)) * noise(x0,q),# Process uncertainty 
        'P': np.eye(len(p0))*p0,# Initial Covariance guess
        'R': np.eye(len(x0))* noise(x0,r),# Measurment uncertainty 
        'H': np.eye(len(df_x.loc[0])),#Observation matrix
        
        }
    

    return Parameter
    
    