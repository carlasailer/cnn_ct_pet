#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:25:39 2019

@author: s1287
"""
import scipy.io as sio

from scripts_interp import swap_axes, filter_upsampled_volume

def interpolate_MATLAB():
    filename = '/home/s1287/no_backup/s1287/results_interp/CT_upsampled_linear.mat'
    CT_upsampled_MATLAB = sio.loadmat(filename)["CT_upsampled"]
    CT_upsampled_MATLAB = list(CT_upsampled_MATLAB[0])
    print('MATLAB file loaded: %s' %filename)
#    with h5py.File('CT_upsampled.mat', 'r') as file:
#        CT_upsampled = [file[element[0]][:] for element in file['CT_upsampled']]
    axis1 = 0
    axis2 = 1
    CT_upsampled_MATLAB_swapped = swap_axes.swap_array_axes(CT_upsampled_MATLAB,axis1, axis2, keyword='MATLAB')
    
   # CT_upsampled_MATLAB_filtered = filter_upsampled_volume.filter_upsampled_volume(CT_upsampled_MATLAB_swapped)

    return CT_upsampled_MATLAB_swapped, CT_upsampled_MATLAB_swapped