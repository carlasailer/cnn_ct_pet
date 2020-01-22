#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:35:41 2019

@author: s1287
"""
import numpy as np
import h5py
import scipy.io as sio
import copy
#from skimage.measure import compare_ssim as ssim
from random import shuffle

def load_data(filenames):
    """ Load data stored in the location specified by parameter
    Parameters:
        filenames     dic of filenames where CT and PET data are stored
    Returns:
        data_CT       list of CT VOIs for each patient
        data_PET      list of PET VOIs for each patient
    """
     #load matlab files 
    with h5py.File(filenames['CT'], 'r') as file:
        data_CT = [file[element[0]][:] for element in file['images_segmented_CT']]
    
    with h5py.File(filenames['PET'], 'r') as file:
        data_PET = [file[element[0]][:] for element in file['images_segmented_PET']]
        
    return data_CT, data_PET

def savePatches(patches_CT, patches_PET, keyword=''):
    """ Save patches to .h5 file
    Parameters:
        patches_CT      (nested) list of CT patches for each patient
        patches_PET     (nested) list of PET patches for each patient
        keyword         string, to be added to the filename
    """
    file_save = 'patches' + keyword
    
    with h5py.File(str(file_save + '.h5'), 'w') as file:
        file.create_dataset('CT', data=patches_CT)
        file.create_dataset('PET', data=patches_PET )
        
#    dic_save = {'patches_CT': patches_CT, 'patches_PET': patches_PET}
 
#    file_save_PET = '/home/s1287/no_backup/s1287/results_interp/PET_patches'
#    CT_array = np.asarray(patches_CT)
#    PET_array = np.asarray(patches_PET)
    
#    dic = {'CT': patches_CT, 'PET': patches_PET}
    #np.savez(file_save, CT_array, PET_array)
    #sio.savemat(file_save,dic)
    

    
def swap_array_axes(data, axis1, axis2, keyword=None):
    """ Swap axes required due to different representations in MATLAB and Python
    Parameters:
        data            list of VOIs for each patient
        axis1, axis2    axes to be flipped
        keyword         can be used for modifications
    Returns:
        data_flipped    list of VOIs for each patient with flipped axes
    """
    data_flipped = []
    #iterate over patients
    for patient in range(1,len(data)+1):
        #some patients should be left out --> only one slice
        if patient in [71,162,180]:
           data_patient = np.swapaxes(data[patient-1],axis1,axis2)
        
        else:
            #create new array with correct flipped dimensions
            if keyword == 'MATLAB':
                data_patient = np.zeros((data[patient-1].shape[2], data[patient-1].shape[0], data[patient-1].shape[1]))
                data_patient = np.swapaxes(data[patient-1], 2,0)
                data_patient = np.swapaxes(data_patient, 1,2)
                    
            else:
                data_patient = np.zeros((data[patient-1].shape[0], data[patient-1].shape[2], data[patient-1].shape[1]))
                #swap axes slicewise
                for slice_nr in range(0, data[patient-1].shape[0]):
                    data_patient[slice_nr,:,:]= np.swapaxes(data[patient-1][slice_nr,:,:],axis1,axis2)
                
        #add patient to a list of flipped arrays
        data_flipped.append(data_patient)
        del data_patient
    
    return data_flipped

def removeZeroPatches(patches_CT, patches_PET):
    """ Remove patches that only contains zeros in the CT from both CT and PET
    Parameters:
        patches_CT      nested list of CT patches for each patient
        patches_PE      nested list of PET patches for each patient
    Returns:
        res_CT          nested list of CT patches without patches with zeros
        res_PET         nested list of PET patches without patches with zeros
    """
    patches_CT_noZero = copy.deepcopy(patches_CT)
    patches_PET_noZero = copy.deepcopy(patches_PET)
    res_CT = [[] for idx in range(0,len(patches_CT))]
    res_PET = [[] for idx in range(0,len(patches_PET))]
    for patient in range(0,len(patches_CT)):
        loc_CT = np.where(np.sum(patches_CT[patient], axis=(1,2,3)) == 0)[0].tolist()
        for idx in loc_CT: 
                patches_CT_noZero[patient][idx] = None
                patches_PET_noZero[patient][idx] = None
        for idx in range(0, len(patches_CT_noZero[patient])):
            if patches_CT_noZero[patient][idx] is not None:
                res_CT[patient].append(patches_CT_noZero[patient][idx])
                res_PET[patient].append(patches_PET_noZero[patient][idx])
                
    return res_CT, res_PET

def splitDataset(patches_CT, patches_PET, ratio):
    """ Arrange the patches randomly and split the dataset according to the given ratio into training and test set
    Parameters:
        patches_CT      nested list of CT patches for each patient
        patches_PET     nested list of PET patches for each patient
        ratio           list containing the ratio for splitting dataset, e.g. ratio = [70, 30]
    Returns:
        training set    list of CT patches and PET patches for training
        test set        list of CT patches and PET patches for training
    """
    #flatten the list of lists
    patches_CT = [item for sublist in patches_CT for item in sublist]
    patches_PET = [item for sublist in patches_PET for item in sublist]
    
    #randomly shuffle the patches - same sort for both CT and PET 
    random_sort = list(range(0,len(patches_CT)))
    shuffle(random_sort)
    #use the created shuffled indices for both CT and PET    
    patches_CT_shuffled = [patches_CT[idx] for idx in random_sort]
    patches_PET_shuffled = [patches_PET[idx] for idx in random_sort]
    
    #split the dataset according to the given ratio
    idx_split = int(np.ceil(ratio[0]/100 *len(random_sort)))
    
    patches_CT_training = patches_CT_shuffled[:idx_split]
    patches_PET_training = patches_PET_shuffled[:idx_split]
    patches_CT_test = patches_CT_shuffled[idx_split:]
    patches_PET_test = patches_PET_shuffled[idx_split:]
    
    return [patches_CT_training, patches_PET_training], [patches_CT_test, patches_PET_test]