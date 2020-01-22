#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:02:35 2018

@author: s1287
"""
import os
import time
import datetime

from scripts_CNN.model_creation import Model_Li1
#from scripts_CNN.data_creation import Dataset
from scripts_CNN.data_creation_meddata import Dataset_MedData
from scripts_CNN.model_training import Trainer
from scripts_CNN.model_evaluation import Evaluator
from scripts_CNN.saver import Saver

def getCNNModel(config, data):
    if config["model"] == 'Li1':
        cnn_model = Model_Li1(config=config, data=data)
    return cnn_model

def run_main(config):
    ##### FILE FOR A SINGLE DATASET FOR FIRST TESTING
    
    ###### processing
    #start the timer
    start_time = time.time()
    #%% create data
    #data = Dataset(config)
    data = Dataset_MedData(config=config)
    
    #%% create model
    cnn_model = getCNNModel(config=config, data=data)
    
    #%% create a results folder
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_folder = os.path.join('/home/s1287/no_backup/s1287/CNN_PET_Prediction_Results/VoxelPatch/', 
                                        str(time_stamp  + '_' + config["model"]))
    #%% train model
    trainer = Trainer(config=config, model=cnn_model.model, data=data, save_folder=save_folder)
    
    ##%% evaluate model
    evaluator = Evaluator(config=config, model=trainer.training_model, data=data, history=trainer.history, 
                          save_folder=trainer.save_folder, training_ssim=trainer.metrics_callback.get_ssim())
    
    #%% save
    Saver(config=config, model=evaluator.evaluation_model, data=data, evaluator=evaluator,
          metrics=evaluator.metrics, trainer=trainer, training_ssim=trainer.metrics_callback.get_ssim())
    
    del cnn_model
     
    #stop the timer
    elapsed_time = time.time() - start_time
    
    print('Reached end of program after: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) )

