#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:26:49 2019

@author: s1287
"""
import numpy as np
from scipy import ndimage, interpolate
import cv2

def interpolate_ndimage(data_CT, data_PET):   
    CT_upsampled = []
    CT_upsampled_filt = []
    
    for patient in range(1,len(data_CT)+1):
        #define zoom factor
        number_ct_images = len(data_CT[patient-1])
        number_pet_images = len(data_PET[patient-1])
        factor = number_pet_images/number_ct_images
        
        CT_volume = data_CT[patient-1] 
        #interpolate along axis 0, round to next integer number --> HU 
        if patient in [71,162,180]:
            CT_upsampled_pat = ndimage.zoom(CT_volume, [1,1])
        else:
            CT_upsampled_pat = ndimage.zoom(CT_volume, [factor,1,1])
        
        #filter
        CT_upsampled_pat_filt = np.copy(CT_upsampled_pat).astype(int).astype('float64')
        for idx in range(0,number_pet_images):
            CT_upsampled_pat_filt[idx][np.where(CT_upsampled_pat[idx] < 10)] = 0
        
        #append to list
        CT_upsampled.append(CT_upsampled_pat)
        CT_upsampled_filt.append(CT_upsampled_pat_filt)
       
    return CT_upsampled, CT_upsampled_filt

#def fill_nan(array):
#    inds = np.arange(array.shape[0])
#    good = np.where(np.isfinite(array))
#    f = interpolate.interp1d(inds[good], array[good], bounds_error=False)
#    B = np.where(np.isfinite(array), array, f(inds))    
#    return B
def interpolate_cv2(data_CT, data_PET):
    patient = 1
    number_ct_images = np.shape(data_CT[patient-1])[0]
    number_pet_images = len(data_PET[patient-1])
    factor = number_pet_images/number_ct_images
   
    current_patient = data_CT[patient-1]
    sizeX = np.shape(current_patient)[1]
    sizeY = np.shape(current_patient)[2]
    sizeZ = number_pet_images
    #dsize = [np.shape(current_patient)[0], np.shape(current_patient[1]), number_pet_images]
    upsampled = cv2.resize(current_patient, (sizeZ, sizeX, sizeY), interpolation=cv2.INTER_LINEAR)
    
def interpolate_interp1d(data_CT, data_PET):
    patient = 1
    number_ct_images = np.shape(data_CT[patient-1])[0]
    number_pet_images = len(data_PET[patient-1])
    
    x = np.linspace(0,20, np.shape(data_CT[patient-1])[1])
    y = np.linspace(0,20, np.shape(data_CT[patient-1])[2])
    z_old = np.linspace(0,20, number_ct_images)
    z_new = np.linspace(0,20, number_pet_images)
     
    V = data_CT[patient-1]
    fn = interpolate.RegularGridInterpolator(x,y,z_new, V)
    pts = fn([x,y,z_new])
    
    return pts
#    for patient in range(1,len(data_CT)+1):
#        CT_upsampled[patient-1] = fill_nan(data_CT[patient-1])
#        print('juu')
        
        
def interpolate_ndimage_nan(data_CT, data_PET):
    #transform zeros to NaN
    data_CT_nan = np.copy(data_CT)
  
    for patient in range(1,len(data_CT)+1):
        data_CT_nan[patient-1][np.where(data_CT[patient-1] == 0)] = np.nan
#        if patient in [71,162,180]:
#            data_CT_nan[patient-1][np.where(data_CT[patient-1] == 0)] = np.nan
#        else:
#           data_CT_nan[patient-1][np.where(data_CT[patient-1] == 0)] = np.nan
##            for slice_nr in range(1,len(data_CT[patient-1])):
#                logical = np.where(data_CT[patient-1][slice_nr]==0)
#                data_CT_nan[patient-1][slice_nr][np.where(data_CT[patient-1]==0)] = np.nan
#   
    #use this data for interpolation --> ndimage not working for NaN's!
    CT_ups_nan = interpolate_cv2(data_CT_nan, data_PET)
    #CT_ups_nan, CT_ups_filt = interpolate_ndimage(data_CT_nan, data_PET)
    
    #transform NaN back to zeros
    CT_ups_zero = np.copy(CT_ups_nan)
    
    for patient in range(1,len(data_CT)+1):
        CT_ups_zero[patient-1][np.where(np.isnan(CT_ups_nan[patient-1]))] = 0
#        if patient in [71,162,180]:
#            CT_ups_zero[patient-1][np.where(np.isnan(CT_ups_nan[patient-1]))] = 0
#        else:
#                CT_ups_zero[patient-1][np.where(np.isnan(CT_ups_nan[patient-1]))] = 0
#    
    return CT_ups_zero, CT_ups_zero
   
                       