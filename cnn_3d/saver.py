#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:02:45 2018

@author: s1287
"""
import os
import datetime
import h5py
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from contextlib import redirect_stdout 
import numpy as np

class Saver:
    def __init__(self, config, model='', data='', history = '', evaluator = '', metrics = '', trainer='', training_ssim=''):
        self.config         = config
        self.save_model     = model
        self.data           = data
        self.history        = history
        self.evaluator      = evaluator
        self.metrics        = metrics
        self.save_folder    = trainer.save_folder
        self.training_ssim  = training_ssim
        self.save()
    
    def save(self):       
        #save the configs
        config_filename = self.save_folder + '/config.txt'
        with open(config_filename, 'w') as f:
            f.write(json.dumps(self.config))
        print('Configs saved.')
        
        #save predicted patches
        self.save_predicted_patches()
        
        #save some patches as subplots
        self.save_predicted_images()

        #save loss curves
        self.saveLossCurves()

        #save ssim curve
        self.save_ssim_curves()
        
        #save the metrics:
        metric_filename = self.save_folder + '/metrics.txt'
        with open(metric_filename, 'w') as f:
            f.write(json.dumps(self.metrics))
        print('Metrics saved.')

        #save the keras model
  #      self.save_model.model.save(str(self.save_folder + '/model.h5'))
        self.save_model.model.save_weights(str(self.save_folder + '/modelWeights.h5'))
        print('Keras model saved.')

        #save the model summary
        summary_file = self.save_folder + '/model_summary.txt'
        with open(summary_file, 'w') as f:
            with redirect_stdout(f):
                self.save_model.model.summary()
        
        del self.save_model
    
    def saveLossCurves(self):
        # create and save loss curves
        loss = np.array(self.metrics['loss'])
        val_loss = np.array(self.metrics['val_loss'])

        plt.figure(figsize=[8, 6])
        plt.plot(loss, 'r', linewidth=3.0)
        plt.plot(val_loss, 'b', linewidth=3.0)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
        #plt.ylim(0, 0.4)
        x_int = []
        locs, labels = plt.xticks()
        for each in locs:
            x_int.append(int(each))
        plt.xticks(x_int)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.savefig(str(self.save_folder + '/fig_loss.png'))
        
        print('Loss curve saved.')
        
    def save_ssim_curves(self):
        """create and save the SSIM over epoch curve
        Parameters:
            -
        Returns:
            -
        """
        plt.figure(figsize=[8,6])
        plt.plot(self.training_ssim, 'r', linewidth=3.0)
        plt.legend('SSIM', fontsize=18)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('SSIM', fontsize=16)
        plt.savefig(str(self.save_folder + '/fig_ssim.png'))
        
        print('SSIM curve saved.')
        
    def save_predicted_images(self):
        # display and save results for the first n patches
        n = 10
        for patch_nr in range(0,n):
            for slice_nr in range(0,11): 
                pred_img = np.squeeze(self.evaluator.y_predict[patch_nr])[slice_nr,:,:]
                label_img = np.squeeze(self.data.test_labels[patch_nr])[slice_nr,:,:]
        
                fig1 = plt.figure()
                fig1.suptitle('Patch %d Slice %d' %(patch_nr, slice_nr))
                
                plt.subplot(1,2,1)
                plt.imshow(label_img)
                plt.title('Ground Truth PET')
                
                plt.subplot(1,2,2)
                plt.imshow(pred_img)
                plt.title('Predicted PET')
                
               #plt.savefig(str(self.save_folder + '/fig_patch%d_slice%d.png' %(patch_nr, slice_nr)))
                plt.savefig(str(self.save_folder + '/fig_patch%d_slice%d.svg' %(patch_nr, slice_nr)))
                plt.close()
                
    def save_predicted_patches(self):
        file_save = self.save_folder + '/predicted_PET_patches'
        with h5py.File(str(file_save + '.h5'), 'w') as file:
            file.create_dataset('predicted_PET_patches', data=self.evaluator.y_predict)