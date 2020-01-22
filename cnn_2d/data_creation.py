from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


class Dataset:
    def __init__(self, config):

        self.config = config
        self.create_data()

    def create_data(self):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        # Find the unique numbers from the train labels
        self.nClasses = len(np.unique(train_labels))

        #preprocess the data
        # Find the shape of input images and create the variable input_shape
        nRows, nCols, nDims = train_images.shape[1:]
        train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)
        test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)

        # Change to float datatype
        train_data = train_data.astype('float32')
        test_data = test_data.astype('float32')

        #mean subtraction:
#        mean_image = np.mean(train_data)
#        train_data_zeroMean = 
#        test_data_zeroMean = 
        #data normalization to lie between 0 and 1
        self.train_data = train_data/255
        self.test_data = test_data/255

        # Change the labels from integer to categorical data 'One-Hot-Coding'
        self.train_labels = to_categorical(train_labels)
        self.test_labels = to_categorical(test_labels)
