#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:02:45 2018

@author: s1287
"""
import os
import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from contextlib import redirect_stdout 
import numpy as np

class Saver:
    def __init__(self, config, model='', data='', history = '', evaluator = '', metrics = '', trainer=''):
        self.config = config
        self.save_model = model
        self.data = data
        self.history = history
        self.evaluator = evaluator
        self.metrics = metrics
        self.save_folder = trainer.save_folder

        self.save()
    
    def save(self):       
        #save the configs
        config_filename = self.save_folder + '/config.txt'
        with open(config_filename, 'w') as f:
            f.write(json.dumps(self.config))
        print('Configs saved.')

        #save the plots
        self.saveLossAndAccuracyCurves()

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
    
    def saveLossAndAccuracyCurves(self):
        # create and save loss and accuracy curves
        loss = np.array(self.metrics['loss'])
        val_loss = np.array(self.metrics['val_loss'])
        acc = np.array(self.metrics['acc'])
        val_acc = np.array(self.metrics['val_acc'])

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
        
        plt.figure(figsize=[8, 6])
        plt.plot(acc, 'r', linewidth=3.0)
        plt.plot(val_acc, 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
       # plt.ylim(0.8, 1.0)
        x_int = []
        locs, labels = plt.xticks()
        for each in locs:
            x_int.append(int(each))
        plt.xticks(x_int)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig(str(self.save_folder + '/fig_accurancy.png'))

        print('Figures saved.')