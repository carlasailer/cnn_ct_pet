#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:28:57 2019

@author: s1287
"""
import numpy as np
from scripts_interp.filtering import filter_manual
from scripts_interp import compare_to_PET

def read_filter_values(filename):
    values = []
    with open(filename, 'r') as f:
        for line in f:
            values.append(line)
        #for line in f.readlines():
            #values.append(line)
    #return only the lines with the slices and 
    return values[0:-1]

def transform_vals(vals):
    patient_nr = int(vals[0])
    
    filter_nr = np.zeros((len(vals)-1, 4))
    
    #process the string and save it to array
    for idx in range(1, len(vals)):
        splitted = vals[idx].split(',')
        if len(splitted) > 3:
            splitted[-1] = splitted[-1].split(';')[0]
            filter_nr[idx-1] = [int(elem) for elem in splitted]
        else:
            splitted[2] = splitted[2].split(';')[0]
            filter_nr[idx-1][0:3] = [int(elem) for elem in splitted]
        
    #filter_nr: first column is slice_nr, 
    #second col is start value for filtering, third col is end value
    return [patient_nr,filter_nr]

if __name__ == '__main__':
    list_patients_no_filt = [3,4,8,13,20] #13 noch fehlerhaft
    last_pat = 39 #202
    patients = list(range(1,last_pat+1))
    for patient in list_patients_no_filt:
        patients.remove(patient)
    
    CT_upsampled = np.load('/home/s1287/no_backup/s1287/results_interp/CT_upsampled_MATLAB.npy')
    data_PET = np.load('/home/s1287/no_backup/s1287/results_interp/data_PET.npy')
    data_CT = np.load('/home/s1287/no_backup/s1287/results_interp/data_CT.npy')
    
    current_patient = 1
    location = '/home/s1287/homeglb/CNN_PET_Prediction/Voxel_Patch_15x15x15/filter_interp/filter_interpolation_%d.txt'  %current_patient
    filter_vals_str = read_filter_values(location)
    filter_vals = transform_vals(filter_vals_str)
    CT_manually_filtered = filter_manual(CT_upsampled, filter_vals)
    
    current_patient = 13
    #patients = [12,14]
    #iterate over patients for manual filtering
    for patient in patients:
            location = '/home/s1287/homeglb/CNN_PET_Prediction/Voxel_Patch_15x15x15/filter_interp/filter_interpolation_%d.txt'  %patient
            filter_vals_str = read_filter_values(location)
            filter_vals = transform_vals(filter_vals_str)
            CT_manually_filtered = filter_manual(CT_manually_filtered, filter_vals)
            
            #compare to PET without filter:
            compare_to_PET.compare_to_PET(data_CT, data_PET, CT_manually_filtered, patient)
            #compare to PET including manual filtering:
            compare_to_PET.compare_to_PET(data_CT, data_PET, CT_manually_filtered, patient, keyword='filt')
            
    np.save('/home/s1287/no_backup/s1287/results_interp/CT_manually_filtered.npy', CT_manually_filtered)
    
    
