# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:09:10 2021

@author: panay
This is a 2-D version of a Kalman filter from scratch
https://www.kalmanfilter.net/kalman1d.html
"""

import numpy as np
# from numpy.random import randint,randn
import plotly.graph_objects as go

class KalmanFilter():
    
    # State Update
    def StateUpdate(self,Estimate_x_n_minus1, K_n, z_n, H):
        '''

        Parameters
        ----------
        Estimate_x_n_minus1 : Vector
            predicted system state vector at time step n−1
        K_n : Matrix
            Kalman Gain
        z_n : Vector
            measurement
        H : Matrix
            observation matrix

        Returns
        -------
        Estimate_x_n : Vector
            estimated system state vector at time step n

        '''
        
        
        Estimate_x_n = Estimate_x_n_minus1 + np.matmul(K_n,(z_n - np.matmul(H, Estimate_x_n_minus1))) 
        
        
        return Estimate_x_n
    
    
    
    # State Extrapolation
    def StateExtrapolation(self,F, Estimate_x_n, G, u_n, w_n):
        '''
        Importand: using Constant Dynamic Model previous state is equal to the next.
        example of this is measuring the height of a building 
        (does not change(constant) within reasonable time)

        Parameters
        ----------
        F : Matrix (n x n)
            state transition matrix
        Estimate_x_n : Vector (n x 1)
            estimated system state vector at time step n
        G : Matrix (n x n)
             control matrix or input transition matrix (mapping control to state variables)
        u : Vector (n x 1)
            control variable or input variable - a measurable (deterministic) input to the system
        w : Vector (n x 1)
             process noise or disturbance - an unmeasurable input that affects the state

        Returns
        -------
       Estimate_x_n_plus1 : Vector
            predicted system state vector at time step n+1 
        
        '''
        w_n= w_n.reshape(len(w_n),1)
        w_n= np.eye(len(w_n)) * w_n
        
        Estimate_x_n_plus1 = np.matmul(F, Estimate_x_n) + np.matmul(G ,u_n) + w_n 
        
        Estimate_x_n_plus1= np.eye(len(Estimate_x_n_plus1)) * Estimate_x_n_plus1
        return Estimate_x_n_plus1
    
    
    # Kalman Gain
    def KalmanGain(self,P_n_minus1, H, R_n):
        '''

        Parameters
        ----------
        P_n_minus1 : Matrix
            prior estimate uncertainty (covariance) matrix of the current sate (predicted at the previous state)
        H : Matrix
            observation matrix
        R_n : Matrix
            Measurement Uncertainty (measurement noise covariance matrix)

        Returns
        -------
        K_n : Matrix
            Kalman Gain

        '''
        
        K_n = np.matmul(P_n_minus1, H.T) / (np.matmul(np.matmul(H, P_n_minus1),H.T) + R_n)
        #handle nan values
        K_n[np.isnan(K_n)] = 0
        return K_n
    
    # Covariance Update
    def CovarianceUpdate (self,I, K_n, H,  P_n_minus1, R_n):
        '''
    
        Parameters
        ----------
        I : Matrix
            Identity matrix
        K_n : Matrix
             Kalman Gain
        H : Matrix
            observation matrix
        P_n_minus1 : Matrix
            prior estimate uncertainty (covariance) matrix of the current sate (predicted at the previous state)
        R_n : Matrix
            Measurement Uncertainty (measurement noise covariance matrix)

        Returns
        -------
        P_n : Matrix
            an estimate uncertainty (covariance) matrix of the current sate

        '''

        P_n = np.matmul(np.matmul((I - np.matmul(K_n, H)), P_n_minus1), (I - np.matmul(K_n,H)).T) + np.matmul(np.matmul(K_n,R_n),K_n.T)
        
        
        return P_n
    
    # Covariance Extrapolation
    def CovarianceExtrapolation(self,F, P_n, Q):
        '''
        
        Parameters
        ----------
        F : Matrix
            state transition matrix
        P_n : Matrix
            estimate uncertainty (covariance) matrix of the current sate
        Q : Matrix
            process noise matrix

        Returns
        -------
        P_n_plus1 : Matrix
            predicted estimate uncertainty (covariance) matrix for the next state

        '''
        
        P_n_plus1 = np.matmul(np.matmul(F, P_n), F.T) + Q
        return P_n_plus1

    def MeasurementEquation(self,H, x_n, R_n):
        '''
        
        Parameters
        ----------
        H : Matrix
            observation matrix
        x_n : Vector
            true system state (hidden state)
        v_n : Vector
             random noise vector
    
        Returns
        -------
        z_n : Vector
            measurement vector
    
        '''
        x_n= x_n.values.reshape(len(x_n),1)
        x_n= np.eye(len(x_n)) * x_n
        
        #also adding noise to the measurement
        z_n = np.matmul(H, x_n) + R_n
        
        return z_n
    
    def Expectation(self,v):
        '''
        
        Parameters
        ----------
        v : Vector
            Error/noise vectors
    
        Returns
        -------
        E : Matrix
            Covariance matrix
            The Expectation of v
    
        '''
        v=list(v)
        mean_v= np.mean(v)
        v = (v - mean_v)**2
        
        E= np.matmul(v,v.T)
        E=np.eye(len(E))*E
        return E
    
    def MeasurementUncertainty(self, u_n):
        '''
        
        Parameters
        ----------
        u_n : Vector
             measurement error
    
        Returns
        -------
        R_n : Matrix
            covariance matrix of the measurement
    
        '''
        #random cant be all the same expecation is then 0 (nxn)
        R_n= self.Expectation(u_n)
        
        return R_n
    
    def Process_noiseUncertainty(self, w_n):
        '''
        
        Parameters
        ----------
        w_n : Vector
             process noise
    
        Returns
        -------
        Q_n : Matrix
             covariance matrix of the process noise
    
        '''
        
        Q_n=self.Expectation(w_n)
        
        return Q_n
    
    def Estimation_Uncertainty(self, x_n,Estimate_x_n):
        '''
        
    
        Parameters
        ----------
        x_n : Vector
             true system state (hidden state)
        Estimate_x_n : Vector
           estimated system state vector at time step n
        
        error : estimation error vector
    
        Returns
        -------
        P_n : Matrix
            covariance matrix of the estimation error
    
        '''
        
        
        error = x_n - Estimate_x_n
       
        P_n= self.Expectation(error)
        
        return P_n
    
    
    #Kalman smoothing
    def Lparameter(self,P_n,F,P_n_plus1):
        '''
        

        Parameters
        ----------
        P_n : Matrix
            variance of the state at n
        F : Matrix
            transition matrix of the state
        P_n_plus1 : Matrix
            variance of the state at n+1

        Returns
        -------
        L : Matrix
            L parameter for backward pass   

        '''
        L = np.matmul(P_n,F) / P_n_plus1
    
        return L
    
    def Smooth_X(self,x_n,L_n,x_n_plus1):
        '''
        

        Parameters
        ----------
        x_n : Matrix
            Estimated x at time n
        L_n : Matrix
            L parameter for Kalman Smoothing
        x_n_plus1 : Matrix
            Estimated x at time n+1

        Returns
        -------
        x_n : Vector
            Smoothed Estimate x at time n 
        

        '''
        x_n = x_n + np.matmul(L_n, (x_n_plus1-x_n))
        
        return x_n
    
    def Smooth_P(self,P_n, L_n, P_n_plus1):
        '''
        

        Parameters
        ----------
        P_n : Matrix
            Estimate P at time n
        L_n : Matrix
            L parameter for Kalman Smoothing
        P_n_plus1 : Matrix
             Estimated P at time n+1

        Returns
        -------
        P_n : Matrix
           Smoothed Estimate P at time n 

        '''
        #make everything to matrix form for calculations
        P_n=np.eye(len(P_n))*P_n
        L_n=np.eye(len(L_n))*L_n
        P_n_plus1=np.eye(len(P_n_plus1))*P_n_plus1
        
        
        P_n = P_n + np.matmul(np.matmul(L_n, (P_n_plus1 - P_n)), L_n.T)
        
        return P_n
    
    
    def plotVariance(self,Y,Y_variance):
            
        fig = go.Figure(data=go.Scatter(
                x=Y.index,
                y=Y,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=Y_variance,
                    visible=True)
            ))
        
        return fig

