# Goal:
# Import Matlab files and create training and test set of voxel patches for both CT and PET

from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import numpy as np
from scripts_patches import visualizations, extract_Patches, helperFunctions
#from scripts_patches import similarityMetrics

def visualize_single_slice(patient, slice_nr):
    image_CT =  data_CT[patient-1][slice_nr,:,:]
    image_PET = data_PET[patient-1][slice_nr,:,:]

    plt.figure()
    plt.imshow(image_CT, cmap=cm.Greys)
    plt.figure()
    plt.imshow(image_PET, cmap=cm.Greys)

#def save_results():
#    np.save('/home/s1287/no_backup/s1287/results_interp/data_CT.npy', data_CT)
#    np.save('/home/s1287/no_backup/s1287/results_interp/data_PET.npy', data_PET)
#    np.save('/home/s1287/no_backup/s1287/results_interp/CT_upsampled_MATLAB.npy', CT_upsampled_MATLAB)
#    np.save('/home/s1287/no_backup/s1287/results_interp/CT_upsampled_filt_MATLAB.npy', CT_upsampled_filt_MATLAB)
#   # np.save('/home/s1287/no_backup/s1287/results_interp/CT_upsampled_PY.npy', CT_upsampled_PY)
#    #np.save('/home/s1287/no_backup/s1287/results_interp/CT_upsampled_filt_PY.npy', CT_upsampled_filt_PY)
#
#def visualize_interpolation(patient):
#    #visualizations.compare_CT(data_CT, data_PET, CT_upsampled_filt_MATLAB, patient, slice_nr)
#    visualizations.compare_to_PET(data_CT, data_PET, CT_upsampled_MATLAB, patient, 'MATLAB_linear')
#    #compare_to_PET.compare_to_PET(data_CT, data_PET, CT_upsampled_PY, patient, 'Python')
#    
if __name__ == '__main__':
    
    #define filenames 
    filenames = {'CT':  '/home/s1287/med_data/Texture/BronchialCarcinoma/CT_segmented_cropped_allimages.mat',
                 'PET': '/home/s1287/med_data/Texture/BronchialCarcinoma/PET_segmented_cropped_allimages.mat'}
    
    #load files
    data_CT, data_PET = helperFunctions.load_data(filenames)
     
    #flip axes to match Matlab representation
    data_CT =  helperFunctions.swap_array_axes(data_CT,  0, 1)
    data_PET = helperFunctions.swap_array_axes(data_PET, 0, 1)
    
    #interpolate missing slices using ndimage
   # CT_upsampled_PY, CT_upsampled_filt_PY = interpolate_ndimage.interpolate_ndimage_nan(data_CT, data_PET)
   # CT_upsampled_MATLAB, CT_upsampled_MATLAB_filt = interpolate_MAT.interpolate_MATLAB()
   
    #visualize interpolation
    patient = 1 # in [1,202]
   # visualize_interpolation(patient)
   
    factor_image = []
    factor_depth = []
    patients_tot = list(range(1,202+1))
    patients_tot.remove(71)
    patients_tot.remove(162)
    patients_tot.remove(180)
#    for patient in patients_tot:
#        size_P = np.shape(data_PET[patient-1])
#        size_C = np.shape(data_CT[patient-1])
#        factor_patient = size_C[1]/size_P[1]
#        factor_image.append(factor_patient)
#        factor_patient = size_C[0]/size_P[0]
#        factor_depth.append(factor_patient)

    #extract patches   
    p_overlap = 0.5
    p_size_PET = [11,5,5]
    p_size_CT = [6,13,13]
    patches_CT, number_CT, patches_PET, number_PET = extract_Patches.manageData(data_CT=data_CT,
                                                                                  data_PET=data_PET, overl=p_overlap, 
                                                                                 size_p_CT=p_size_CT, size_p_PET=p_size_PET)
    
    #remove patches with only zeros
    patches_CT_noZero, patches_PET_noZero = helperFunctions.removeZeroPatches(patches_CT, patches_PET)
    
    
 #   patch_range = [0,1]
#    visualizations.compare_patches(patches_CT_noZero[patient-1], patches_PET_noZero[patient-1], patient, patch_range)
    
 
    #calculate structural similarity index for each patch and sort into training and test set
#    similarityMetrics.calcMutualInformation(patches_CT_noZero, patches_PET_noZero)

    #random arrange of patches & split dataset with a given ratio
    ratio = [70, 30]
    training_set, test_set = helperFunctions.splitDataset(patches_CT_noZero, patches_PET_noZero, ratio)
    print('Finished creation of training and test set.')
  
    #save patches
    helperFunctions.savePatches(training_set[0], training_set[1], '_training')
    helperFunctions.savePatches(test_set[0], test_set[1], '_test')
    