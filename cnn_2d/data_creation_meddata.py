from keras.utils import to_categorical
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Dataset_MedData:
    def __init__(self, config, dataset_nr):
        self.config = config
        self.dataset_nr = dataset_nr
        
        self.create_data()

    def create_data(self):
        #specify path and import first dataset
        contents = os.listdir(self.config["input_path"])
        filename = self.config["input_path"] + contents[self.dataset_nr]
        f = h5py.File(filename, 'r')
        print(filename)
        
        #extract the data and change to double precession
        train_data = np.array(f['X_train']).astype('float32')
        train_labels = np.array(f['Y_train'])
        test_data = np.array(f['X_test']).astype('float32')
        test_labels = np.array(f['Y_test'])
                
        #preprocess labels:
        #make binary labels
        # 10 = low suv, 11 = high suv
        self.train_data, self.train_labels =   self.removeClassFromDataset_binary(train_data, train_labels, 0)
        self.test_data, self.test_labels =     self.removeClassFromDataset_binary(test_data, test_labels, 0)

        #count the samples for each class
        self.n_lowsuv_train = len([label_item for label_item in train_labels if label_item != 0])
        self.n_highsuv_train = len([label_item for label_item in train_labels if label_item != 1])
        self.n_lowsuv_test = len([label_item for label_item in test_labels if label_item != 0])
        self.n_highsuv_test = len([label_item for label_item in test_labels if label_item != 1])
        
        self.prob_low_suv = self.n_lowsuv_train/(self.n_lowsuv_train + self.n_highsuv_train)
        self.prob_high_suv = self.n_highsuv_train/(self.n_lowsuv_train + self.n_highsuv_train)
        
        print('Percentage of training samples in class "LOW_SUV": ', str(self.prob_low_suv))
        print('Percentage of training samples in class "HIGH_SUV": ', str(self.prob_high_suv))
        print('Percentage of test samples in class "LOW_SUV": ', str(self.n_lowsuv_test/(self.n_lowsuv_test + self.n_highsuv_test)))
        print('Percentage of test samples in class "HIGH_SUV": ', str(self.n_highsuv_test/(self.n_lowsuv_test + self.n_highsuv_test)))

        self.nClasses = len(np.unique(train_labels))
        
#        plt.figure()
#        plt.imshow(train_data[248])
#        plt.show()
        
        #preprocess the images:
        if self.config["mean_subtrac"] == True:
            self.train_data, self.test_data = self.performMeanSubtraction(self.train_data, self.test_data)

        #scale training and test data to a specific range
        if self.config["scaling"] == True:
            max_val = 1
            self.train_data, self.test_data = self.scaleToRange(self.train_data, self.test_data, (0,max_val))
        
        #Create a balanced dataset
        if self.config["small_dataset"] == True:
            self.train_data, self.train_labels = self.createBalancedSampleDataset(self.train_data, self.train_labels, 'train')
            #self.test_data, self.test_labels = self.createBalancedSampleDataset(self.test_data, self.test_labels, 'test')
        
        # count the samples for each class
        self.n_lowsuv_train = len([label_item for label_item in self.train_labels if label_item != 0])
        self.n_highsuv_train = len([label_item for label_item in self.train_labels if label_item != 1])
        self.n_lowsuv_test = len([label_item for label_item in self.test_labels if label_item != 0])
        self.n_highsuv_test = len([label_item for label_item in self.test_labels if label_item != 1])

        self.prob_low_suv = self.n_lowsuv_train / (self.n_lowsuv_train + self.n_highsuv_train)
        self.prob_high_suv = self.n_highsuv_train / (self.n_lowsuv_train + self.n_highsuv_train)

        print('Percentage of training samples in class "LOW_SUV": ', str(self.prob_low_suv))
        print('Percentage of training samples in class "HIGH_SUV": ', str(self.prob_high_suv))
        print('Percentage of test samples in class "LOW_SUV": ', str(self.n_lowsuv_test / (self.n_lowsuv_test + self.n_highsuv_test)))
        print('Percentage of test samples in class "HIGH_SUV": ', str(self.n_highsuv_test / (self.n_lowsuv_test + self.n_highsuv_test)))

        #use one-hot-coding
        if self.config["loss_function"] == 'categorical_crossentropy':
            self.train_labels = to_categorical(self.train_labels)
            self.test_labels = to_categorical(self.test_labels)
        else:
            self.train_labels = to_categorical(self.train_labels)
            self.test_labels = to_categorical(self.test_labels)
            
        # Find the shape of input images and create the variable input_shape
        self.train_data = self.adaptDimensions(dataset=self.train_data)
        self.test_data = self.adaptDimensions(dataset=self.test_data)
        
        #to be deleted later
        train_data = self.train_data
        train_labels = self.train_labels
        test_data = self.test_data
        test_labels = self.test_labels
        print('Preprocessing finished.')

    def adaptDimensions(self, dataset, ndims=1):
        nRows, nCols = dataset.shape[1:]
        dataset_shaped = dataset.reshape(dataset.shape[0], nRows, nCols, ndims)
        
        return dataset_shaped
    
    def removeClassFromDataset_binary(self, dataset, labels, class_value):
        #only use the datapoints where the label is unequal the given class_value and assigns binary value
        labels_filtered = [label_item for label_item in labels if label_item != class_value]
        dataset_filtered = np.array([data_item for data_item, label_item in zip(dataset, labels) if label_item != class_value])
        
        #make binary: 0: SUV = 10, 1: SUV = 11
        for index in range(len(labels_filtered)):
            if labels_filtered[index][0] == 10:
                labels_filtered[index][0] = 0
            else:
                labels_filtered[index][0] = 1

        # convert back to np.array
        labels_filtered = np.array(labels_filtered)
        # labels are either 0 or 1
        return dataset_filtered, labels_filtered

    def removeClassFromDataset(self, dataset, labels, class_value):
        # only use the datapoints where the label is unequal the given class_value and assigns binary value
        labels_filtered = np.array([label_item for label_item in labels if label_item != class_value])
        dataset_filtered = np.array([data_item for data_item, label_item in zip(dataset, labels) if label_item != class_value])
        
        # labels are either 10 or 11
        return dataset_filtered, labels_filtered
    
    def performMeanSubtraction(self, train_data, test_data):
        #calculate the mean image of train_data set:
        current_mean = np.mean(train_data, axis=0)
        
        #subtract this mean from both train_data and test_data
        train_data_meanSub = train_data - current_mean
        test_data_meanSub = test_data - current_mean
        
        return train_data_meanSub, test_data_meanSub
    
    def scaleToRange(self, train_data, test_data, scaling_range):
        maximum = np.max(train_data, axis=0)
        minimum = np.min(train_data, axis=0)
        
        train_data_scaled = (scaling_range[1]-scaling_range[0]) * (train_data - minimum)/(maximum - minimum) + scaling_range[0]
        test_data_scaled = (scaling_range[1]-scaling_range[0]) * (test_data - minimum)/(maximum - minimum) + scaling_range[0]
        
        return train_data_scaled, test_data_scaled

    def createBalancedSampleDataset(self, train_data, train_labels, set_type):
        if set_type == 'train':
            number_samples_per_class = min(self.n_lowsuv_train, self.n_highsuv_train)
            print('Samples in train:', str(number_samples_per_class))
        elif set_type == 'test':
            number_samples_per_class = min(self.n_lowsuv_test, self.n_highsuv_test)
            print('Samples in test:', str(number_samples_per_class))
        #number_samples_per_class = 3
        
        chosen_data = []
        chosen_labels = []
        samples_low_suv = []
        samples_high_suv = []
        
        #separate high and low suv samples:
        samples_low_suv =  [data_item for data_item, label_item in zip(train_data, train_labels) if label_item != 0]
        samples_high_suv = [data_item for data_item, label_item in zip(train_data, train_labels) if label_item != 1]
        
        #create dataset where low and high suv alternate:
        for counter in range(0, number_samples_per_class):
            chosen_data.append(samples_low_suv[counter])
            chosen_labels.append(0)
            chosen_data.append(samples_high_suv[counter])
            chosen_labels.append(1)
        
        train_data_sampled = np.asarray(chosen_data).astype('float32')
        train_labels_sampled = np.asarray(chosen_labels).astype('uint8')
        print(train_labels_sampled)
        
        return train_data_sampled, train_labels_sampled