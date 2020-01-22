#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:16:04 2019

@author: s1287
"""
import keras 
import keras.backend as K
import numpy as np

def calc_ssim_git(y_true, y_pred):
    """structural similarity measurement system."""
    ## K1, K2 are two constants, much smaller than 1
    K1 = 0.04
    K2 = 0.06
    
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5

    ## L, number of pixels, C1, C2, two constants
    L =  33
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim 

def calc_ssim_git_np(y_true, y_pred):
    """structural similarity measurement system."""
    ## K1, K2 are two constants, much smaller than 1
    K1 = 0.04
    K2 = 0.06
    
    ## mean, std, correlation
    mu_x = np.mean(y_pred)
    mu_y = np.mean(y_true)
    
    sig_x = np.std(y_pred)
    sig_y = np.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5

    ## L, number of pixels, C1, C2, two constants
    L =  33
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim 

def calc_ssim(y_true, y_pred):
        """ compares two images by calculating the structured similarity (SSIM)
        Parameters:
            y_true  ground truth image 
            y_pred  predicted image
        Returns:
            SSIM    structured similarity value for the two images
        """
        ## K1, K2 are two constants, much smaller than 1
        K1 = 0.04
        K2 = 0.06
        
        ## mean, std, correlation
        mu_x = np.mean(y_pred)
        mu_y = np.mean(y_true)
        
        sig_x = np.std(y_pred)
        sig_y = np.std(y_true)
        sig_xy = (sig_x * sig_y) ** 0.5
    
        ## L, number of pixels, C1, C2, two constants
        # L is the dynamic range of the pixel values = maximum value
        # maybe perform a range mapping?!?!
        L =  198685
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
    
        ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
#        
        return ssim 


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
                
    def on_train_begin(self, logs={}):
        self.ssims = []
        self.losses = []
   
    def on_train_end(self, logs={}):
        print('Final SSIM: %f percent' %self.ssims[-1])
        return         
    
    def on_epoch_begin(self, epoch, logs={}):
        return
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])
        
        ssim = calc_ssim(y_true, y_pred) *100
        self.ssims.append(ssim)
        print('SSIM for epoch %d is %f percent' %(epoch, ssim))
        
#         
    def get_ssim(self):
        return self.ssims      
        