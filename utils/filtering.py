#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:51:13 2019

@author: s1287
"""
import numpy as np

def filter_manual(data, filter_vals):
    patient = filter_vals[0]
    vals = filter_vals[1]
    data_filt = np.copy(data)
    
    for slice_nr in range(int(vals[0][0]), int(vals[-1][0]+1)):
        lower = vals[slice_nr][1]
        #upper = vals[slice_nr][2]
        data_filt[patient-1][slice_nr][np.where(data_filt[patient-1][slice_nr] < lower)] = 0
        #data[patient-1][slice_nr][np.where(data[patient-1][slice_nr] > upper)] = 0
        
        if vals[slice_nr][3] != 0.0:
            middle = int(vals[slice_nr][3])
            x_vals = np.shape(data_filt[patient-1][slice_nr])[0]
            y_vals = np.shape(data_filt[patient-1][slice_nr])[1]
            
            for x,y in zip(range(0,x_vals), range(0,y_vals)):
                    current_datapoint = int(round(data_filt[patient-1][slice_nr][x][y]))
                    if current_datapoint in range(middle-75, middle+75):
                        data_filt[patient-1][slice_nr][x][y] = 0
    
    return data_filt