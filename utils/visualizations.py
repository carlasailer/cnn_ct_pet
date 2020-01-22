#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:50:16 2019

@author: s1287
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave

def compare_to_CT(data_CT, data_PET, CT_upsampled_filt_MATLAB, patient, slice_nr):
    #visualize slices after upsamling
    number_ct_images = len(data_CT[patient-1])
    number_pet_images = len(data_PET[patient-1])
    slice_nr_old = slice_nr
    slice_nr_new =  np.round(slice_nr*number_pet_images/number_ct_images).astype(int)
    
    fig,(ax1, ax2) = plt.subplots(1,2)
    title_old = 'Original CT slice: ' + str(slice_nr_old)
    ax1.set_title(title_old)
    ax1.imshow(data_CT[patient-1][slice_nr_old])#, cmap=cm.Greys)
    
    title_new = 'Corresponding New CT slice: ' + str(slice_nr_new)
    ax2.set_title(title_new)
    ax2.imshow(CT_upsampled_filt_MATLAB[patient-1][slice_nr_new,:,:])

    save_name = '/home/s1287/no_backup/s1287/results_interp/to_CT/originial_patient%d_slice%d.png' % (patient, slice_nr_old)
    fig.savefig(save_name)
    plt.close('all')
    
def compare_to_PET(data_CT, data_PET, CT_upsampled_filt_MATLAB, patient, keyword=''):
    def compare_to_one(slice_nr):
        fig,(ax1, ax2) = plt.subplots(1,2)
        title_PET = 'Original PET slice: ' + str(slice_nr)
        ax1.set_title(title_PET)
        ax1.imshow(data_PET[patient-1][slice_nr])
        
        title_CT = 'New CT slice: ' + str(slice_nr)
        ax2.set_title(title_CT)
        ax2.imshow(CT_upsampled_filt_MATLAB[patient-1][slice_nr,:,:])
        
        if not keyword:
            #if string is empty
            save_name = '/home/s1287/no_backup/s1287/results_interp/to_PET/wPET_patient%d_slice%d.png' %(patient, slice_nr)
        elif keyword:
            #if string is not empty
            save_name = '/home/s1287/no_backup/s1287/results_interp/to_PET/wPET_patient%d_slice%d_%s.png' %(patient, slice_nr, keyword)
           
        fig.savefig(save_name)
        plt.close('all')

    for slice_number in range(0, data_PET[patient-1].shape[0]):
        compare_to_one(slice_number)

def compare_patches(pat_CT, pat_PET, patient, range_pat):
    for patch in range(range_pat[0], range_pat[1]):
        for slice_nr in range(0,np.shape(pat_CT[patch])[0]):
            img = pat_CT[patch][slice_nr,:,:]
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(img)
            title_CT = 'CT: Patch %d Slice %d Patient %d' %(patch,slice_nr,patient)
            plt.title(title_CT)
            save_name = '/home/s1287/no_backup/s1287/results_interp/patches_CT/Patient%d_Patch%d_Slice%d.png' %(patient,patch,slice_nr)
            fig.savefig(save_name)
            #imsave(save_name, img)
            plt.close()
        
        for slice_nr in range(0,np.shape(pat_PET[patch])[0]):
            img = pat_PET[patch][slice_nr,:,:]
            fig = plt.figure()
            plt.axis('off') 
            plt.imshow(img)
            title_PET = 'PET Patch %d Slice %d Patient %d' %(patch,slice_nr,patient)
            plt.title(title_PET)
            save_name = '/home/s1287/no_backup/s1287/results_interp/patches_PET/Patient%d_Patch%d_Slice%d.png' %(patient,patch,slice_nr)
            fig.savefig(save_name)
            plt.close()