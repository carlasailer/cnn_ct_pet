from keras.utils import to_categorical
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Dataset_MedData:
    def __init__(self, config):
        self.config = config
        
        self.create_data()

    def create_data(self):
        #specify path and import first dataset
        contents = os.listdir(self.config["input_path"])
        filename_test = self.config["input_path"] + contents[0]
        filename_training = self.config["input_path"] + contents[1]
        
        with h5py.File(filename_training, 'r') as file:
            self.training_CT = np.array(file.get('CT'))
            self.training_PET = np.array(file.get('PET'))
            
        with h5py.File(filename_test, 'r') as file:
            self.test_CT = np.array(file.get('CT'))
            self.test_PET = np.array(file.get('PET'))
            
        self.train_data = self.training_CT
        self.train_labels = self.training_PET
        self.test_data = self.test_CT
        self.test_labels = self.test_PET
        
#        #scale training and test data to a specific range
#        if self.config["scaling"] == True:
#            max_val = 1
#            self.train_data, self.test_data = self.scaleToRange(self.train_data, self.test_data, (0,max_val))
#        
        # Find the shape of input images and create the variable input_shape
        self.train_data     =   self.adaptDimensions(dataset=self.train_data)
        self.train_labels   =   self.adaptDimensions(dataset=self.train_labels)
        self.test_data      =   self.adaptDimensions(dataset=self.test_data)
        self.test_labels    =   self.adaptDimensions(dataset=self.test_labels)
        
#        #reduce training set 
#        ratio = 0.05
#        self.train_data = self.train_data[:int(np.floor(ratio*len(self.train_data)))]
#        self.train_labels = self.train_labels[:int(np.floor(ratio*len(self.train_labels)))]
#        self.test_data = self.test_data[:int(np.floor(ratio*len(self.test_data)))]
#        self.test_labels = self.test_labels[:int(np.floor(ratio*len(self.test_labels)))]
#        #to be deleted later
#        train_data = self.train_data
#        train_labels = self.train_labels
#        test_data = self.test_data
##        test_labels = self.test_labels
        print('Preprocessing finished.')

    def adaptDimensions(self, dataset, ndims=1):
        dataset_shaped = dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], ndims)
        
        return dataset_shaped
      
    def performMeanSubtraction(self, train_data, test_data):
        #calculate the mean image of train_data set:
        current_mean = np.mean(train_data, axis=0)
        
        #subtract this mean from both train_data and test_data
        train_data_meanSub = train_data - current_mean
        test_data_meanSub = test_data - current_mean
        
        return train_data_meanSub, test_data_meanSub
    