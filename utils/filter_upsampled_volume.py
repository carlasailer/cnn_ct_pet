#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:22:30 2019

@author: s1287
"""
import numpy as np

def filter_upsampled_volume(data):
    data_filt = []
    for patient in range(1, len(data)+1):
        pat_filt = np.copy(data[patient-1]).astype(int).astype('float64')
        for idx in range(0,np.shape(pat_filt)[0]):
            pat_filt[idx][np.where(data[patient-1][idx] < 10)] = 0#
            
        data_filt.append(pat_filt)
        
    return data_filt