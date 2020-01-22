#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:25:02 2019

@author: s1287
"""
import h5py
import os
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

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim 

def calc_ssim(y_true, y_pred):
    """Calculates the structured similarity of two images, ssim is in the range [-1,1]
    Parameters: 
        y_true       voxel used for calculation of SSIM
        y_pred       voxel used for calculation of SSIM
    Returns:
        ssim_value   value of the structured similarity between the two images
    """
#    size = y_true.shape
#    print('The shape is:')
#    print(size)
    single_ssim = []
    try:
        for slice_nr in range(0, y_true.shape[0]):
     #   slice_ssim = compare_ssim(y_true[slice_nr,:,:], y_pred[slice_nr,:,:], win_size=3)
            slice_ssim = compare_ssim(y_true[slice_nr,:,:], y_pred[slice_nr,:,:], win_size=3, gaussian_weights=True)
            single_ssim.append(slice_ssim)
        ssim_mean = np.mean(single_ssim)
        
    except IndexError:
        ssim_mean = 0
        
    
    return ssim_mean
    
#def calc_ssim_multichannel (y_true, y_pred):#
#    return compare_ssim(y_true, y_pred, multichannel=True, win_size=3)

def ssim_fct(y_true, y_pred):
    """wrapper function to fit into the Keras framework
    Parameters:
        y_true      ground truth voxel
        y_pred      voxel predicted by network
    Returns:
        ssim        value of the structural similarity, suited as loss function
    """
    def ssim(y_true, y_pred):
        return -calc_ssim(K.squeeze(y_true), K.squeeze(y_pred))
    return ssim
    

if __name__ == '__main__':
    contents = os.listdir('/home/s1287/no_backup/s1287/results_interp/patches_for_CNN/')
    filename_test = '/home/s1287/no_backup/s1287/results_interp/patches_for_CNN/' + contents[0]
    filename_training = '/home/s1287/no_backup/s1287/results_interp/patches_for_CNN/' + contents[1]
        
    with h5py.File(filename_training, 'r') as file:
        training_CT = np.array(file.get('CT'))
        training_PET = np.array(file.get('PET'))
        
    with h5py.File(filename_test, 'r') as file:
        test_CT = np.array(file.get('CT'))
        test_PET = np.array(file.get('PET'))
        
    train_data = training_CT
    train_labels = training_PET
    test_data = test_CT
    test_labels = test_PET
    
    example_PET1 = train_labels[0]
    example_PET2 = train_labels[1]
    
    current_ssim = calc_ssim(example_PET1, example_PET2)
    current_ssim1 = calc_ssim_multichannel(example_PET1, example_PET2)
    print(current_ssim)
    print('SSIM Multichannel %d' %current_ssim1)
