#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:01:56 2019

@author: s1287
"""

from keras.preprocessing.image import ImageDataGenerator

class Data_Augmentation:
    def __init__(self):#, train_data, train_labels):
#        self.train_data = train_data
#        self.train_labels = train_labels
        self.augment_data()
    
    def augment_data(self):
        self.dataset_gen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.3,
                height_shift_range=0.3,
                #   rescale=1./255,
                #shear_range=0.2,
               # zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

        print('Data augmentation performed.')
