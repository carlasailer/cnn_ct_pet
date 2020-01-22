#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:02:35 2018

@author: s1287
"""
import os
import time
import datetime

from model_creation import *#Model_FirstCNN, Model_CUI_CNN3, Model_CUI_CNN, Model_CUI_CNN3_easy, Model_Truong, Model_NN
from data_creation import Dataset
from main_multiple import getCNNModel
from data_creation_meddata import Dataset_MedData
from model_training import Trainer
from model_evaluation import Evaluator
from saver import Saver

def run_main_single(config):
    ##### FILE FOR A SINGLE DATASET FOR FIRST TESTING
    
    ###### processing
    #start the timer
    start_time = time.time()
    #%% create data
    #data = Dataset(config)
    data = Dataset_MedData(config=config, dataset_nr=config["dataset_nr"])
    
    #%% create model
    cnn_model = getCNNModel(config=config, data=data)
    
    #%% create a results folder
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_folder = os.path.join('/home/s1287/no_backup/s1287/CNN_PET_Prediction_Results/', 
                                        str(time_stamp  + '_' + config["model"]+ '_single_'))
    #%% train model
    trainer = Trainer(config=config, model=cnn_model.model, data=data, save_folder=save_folder)
    
    ##%% evaluate model
    evaluator = Evaluator(config=config, model=trainer.training_model, data=data, history=trainer.history)
    
    #%% save
    Saver(config=config, model=evaluator.evaluation_model, data=data, evaluator=evaluator, metrics=evaluator.metrics, trainer=trainer)
    
    del cnn_model
     
    #stop the timer
    elapsed_time = time.time() - start_time
    
    print('Reached end of program after: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) )
