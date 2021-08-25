# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:55:53 2021

@author: panay
"""
from Kalman_2D_scratch import KalmanFilter
from Kalmanparameters import get_Parameters

import numpy as np
import pandas as pd
import plotly.graph_objects as go

#get the data
df=pd.read_csv(r'./Simulated_Values.csv')
#have the Y values here
df_x=df.drop('Y_random',axis=1)
#df_x=df_x.drop('X_2',axis=1)
#df_x=df_x.drop('Y',axis=1)
df_y=df[['Y']]


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
  
#get Kalman stuff
Kf= KalmanFilter
Parameter = get_Parameters(df_x,df_y,q,r)
# identity matrix
I = np.eye(len(Parameter.get('H')))

#lists to save things

True_Y=[]
Ys=[]
Ys_Variances=[]

Predictions=[]
Pred_Variances=[]

Predictions1=[]
Pred_Variances1=[]

Predictions2=[]
Pred_Variances2=[]

Kalmangain=[]

Measurements=[]
Measurements2=[]

Current_State_Estimate=[]
Current_State_Variance=[]

Current_State_Estimate2=[]
Current_State_Variance2=[]


#Begin Kalmanfilter
Predicted_Estimate=Parameter.get('x0')
Predicted_Variance=Parameter.get('P')

#get predictions for n+1
Predicted_Estimate = Kf.StateExtrapolation(Kf,
                                           Parameter.get('F'),
                                           Predicted_Estimate,
                                           Parameter.get('G'),
                                           Parameter.get('u'),
                                           np.diag(Parameter.get('Q'))) # want only the w noise vector

Predicted_Variance = Kf.CovarianceExtrapolation(Kf,
                                                Parameter.get('F'),
                                                Predicted_Variance,
                                                Parameter.get('Q'))

for i in range(len(df.index)):
    
    #save first because of initial guess, will save next state estimate after loop
    Predictions.append(np.diag(Predicted_Estimate))
    Pred_Variances.append(np.diag(Predicted_Variance))
    
    Predictions1.append(np.diag(Predicted_Estimate)[0])
    Pred_Variances1.append(np.diag(Predicted_Variance)[0])
    
    Predictions2.append(np.diag(Predicted_Estimate)[1])
    Pred_Variances2.append(np.diag(Predicted_Variance)[1])
    
    #save the true value with noise added, using only y as the measurement
    measurement = Kf.MeasurementEquation(Kf,
                                         Parameter.get('H'),
                                         df_x.loc[i],
                                         Parameter.get('R'))
    Measurements.append(np.diag(measurement)[0])
    Measurements2.append(np.diag(measurement)[1])
    
    
    K = Kf.KalmanGain(Kf,
                      Predicted_Variance,
                      Parameter.get('H'),
                      Parameter.get('R'))
    
    Kalmangain.append(K)
    
    #update for the true value (create the joint distribution of x_t and y_t)
    Predicted_Estimate = Kf.StateUpdate(Kf,
                                        Predicted_Estimate,
                                        K,
                                        measurement,
                                        Parameter.get('H'))
    
    Predicted_Variance = Kf.CovarianceUpdate(Kf,
                                             I,
                                             K,
                                             Parameter.get('H'),
                                             Predicted_Variance,
                                             Parameter.get('R'))
    
    Current_State_Estimate.append(np.diag(Predicted_Estimate)[0])
    Current_State_Variance.append(np.diag(Predicted_Variance)[0])
    
    Current_State_Estimate2.append(np.diag(Predicted_Estimate)[1])
    Current_State_Variance2.append(np.diag(Predicted_Variance)[1])
    
    #get predictions for the joint distribution of (joint distribution of x_t+1 and y_t+1)
    Predicted_Estimate = Kf.StateExtrapolation(Kf,
                                               Parameter.get('F'),
                                               Predicted_Estimate,
                                               Parameter.get('G'),
                                               Parameter.get('u'),
                                               np.diag(Parameter.get('Q')))
    
    Predicted_Variance = Kf.CovarianceExtrapolation(Kf,
                                                    Parameter.get('F'),
                                                    Predicted_Variance,
                                                    Parameter.get('Q'))
    
    #The predicted y, in this case, is the last variable of the Predicted state (i.e. Predicted_Estimate)
    Y = np.diag(Predicted_Estimate)[-1]
    Y_Var = np.diag(Predicted_Variance)[-1]

    Ys.append(Y)
    Ys_Variances.append(Y_Var)
    
    True_Y.append(df['Y'].loc[i])
    
    #Re-initialise the (parameters) noise
    Parameter = get_Parameters(df_x,df_y,q,r)

# Kalman Smoother
#Kalman filter results
Results=pd.DataFrame({'Predictions':Predictions,
                      'Prediction Variances':Pred_Variances,
                      'Predictions x1':Predictions1,
                      'Prediction Variances x1':Pred_Variances1,
                      'Predictions x2':Predictions2,
                      'Prediction Variances x2':Pred_Variances2,
                      'Kalman Gain':Kalmangain,
                      'Measurements x1':Measurements,
                      'Measurements x2':Measurements2,
                      'State Estimate x1':Current_State_Estimate,
                      'State Variance x1':Current_State_Variance,
                      'State Estimate x2':Current_State_Estimate2,
                      'State Variance x2':Current_State_Variance2,
                      'Y':Ys,
                      'Y_Variance':Ys_Variances
                      })

fig =go.Figure([
go.Scatter(
    name='Measurement x1',
    x=Results.index,
    y= Results['Measurements x1'],
    mode='lines'
   
),
go.Scatter(
    name='Measurement x2',
    x=Results.index,
    y= Results['Measurements x2'],
    mode='lines'
    
),
go.Scatter(
    name='Predictions x1',
    x=Results.index,
    y=Results['Predictions x1'],
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=Results['Prediction Variances x1'],
            visible=True)
),
go.Scatter(
    name='Predictions x2',
    x=Results.index,
    y=Results['Predictions x2'],
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=Results['Prediction Variances x2'],
            visible=True)
)
])

fig.write_html("./test_Predictions.html")


fig=Kf.plotVariance(Kf,Results['Y'], Results['Y_Variance'])
   


fig = fig.add_scatter(y=True_Y,mode='markers')

fig.write_html("./test_Y with variance.html")


# Kalman Smoother
Smoothed_Pred=[]
Smoothed_Var=[]

Smoothed_Pred_Y=[]
Smoothed_Var_Y=[]

for i in reversed(range(len(Results.index))):
    
    L = Kf.Lparameter(Kf,
                      Results['Prediction Variances'][i-1],
                      Parameter.get('F'),
                      Results['Prediction Variances'][i])
    
    Predicted_Estimate = Kf.Smooth_X(Kf,
                                     Results['Predictions'][i-1],
                                     L,
                                     Results['Predictions'][i])
    
    Smoothed_Pred.append(Predicted_Estimate)
    Smoothed_Pred_Y.append(Predicted_Estimate[-1])
    
    Pred_Variances = Kf.Smooth_P(Kf,
                                 Results['Prediction Variances'][i-1],
                                 L,
                                 Results['Prediction Variances'][i])
    
    Smoothed_Var_Y.append(np.diag(Pred_Variances)[-1])
    Smoothed_Var.append(np.diag(Pred_Variances))
    #to stop the backward pass at 0
    if i == 1:
        break



pd.options.plotting.backend = "plotly"
Smoothed_Results=pd.DataFrame({'Smoothed Predictions':Smoothed_Pred,
                               'Smoothed Predictions Y':Smoothed_Pred_Y,
                               'Smoothed Variances':Smoothed_Var,
                      'Smoothed Variances Y':Smoothed_Var_Y
                      
                      })
#reverse the Smoothed results to plot
Smoothed_Results=Smoothed_Results.iloc[::-1]
Smoothed_Results=Smoothed_Results.reset_index()
Smoothed_Results=Smoothed_Results.drop('index', axis=1)
Smoothed_Results=Smoothed_Results.drop(0)
Smoothed_Results['Y']=df_y['Y']


fig =go.Figure([
    
    go.Scatter(
        name='Y',
        x=Smoothed_Results.index,
        y= Smoothed_Results['Y'],
        mode='lines'
   
    ),
    
    
    go.Scatter(
        name='Smoothed Predictions',
        x=Smoothed_Results.index,
        y=Smoothed_Results['Smoothed Predictions Y'],
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=Smoothed_Results['Smoothed Variances Y'],
                visible=True)
        
        ),
    go.Scatter(
        name='Predicted Y',
        x=Results.index,
        y=Results['Y'],
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=Results['Y_Variance'],
            visible=True)
    )

]
)


fig.write_html("./Smoothed_Predictions.html")

    
def get_Smoothed_Results():
    return Smoothed_Results
    
    