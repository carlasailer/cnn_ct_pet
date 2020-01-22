#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:02:43 2018

@author: s1287
"""
import os
import time
import datetime
from pathlib import Path

from model_creation import Model_CUI_CNN3, Model_Sharma, Model_Sharma_addConv, Model_Sharma_addConv2 #Model_FirstCNN, Model_CUI_CNN3, Model_CUI_CNN, Model_CUI_CNN3_easy, Model_Truong, Model_NN
from data_creation import Dataset
from data_creation_meddata import Dataset_MedData
from model_training import Trainer
from model_evaluation import Evaluator
from saver import Saver

from MetricsStatistics import calculate_metricsStatistics


def run_main_multiple(config):    
    ###### FILE FOR ALL FOLDERS! 
          
    ###### processing
    #start the timer
    start_time = time.time()
    metrics_list = []

    # %% create a results folder
    results_folder = create_results_folder(config["model"])

    for folder_nr in config["dataset_nr"]:
        #%% create data
        data = Dataset_MedData(config=config, dataset_nr=folder_nr)
        
        #%% create model
        cnn_model = getCNNModel(config, data)

        #%% train model
        trainer = Trainer(config=config, model=cnn_model.model, data=data, save_folder=results_folder)
    
        ##%% evaluate model
        evaluator = Evaluator(config=config, model=trainer.training_model, data=data, history=trainer.history)
    
        #%% save
        saver = Saver(config=config, model=evaluator.evaluation_model, data=data, evaluator=evaluator, metrics=evaluator.metrics, trainer=trainer)
        metrics_list.append(evaluator.metrics)
        del cnn_model
    #%%
    #stop the timer
    elapsed_time = time.time() - start_time
    
    calculate_metricsStatistics(metrics=metrics_list, save_folder=results_folder)
    
    print('Reached end of program after: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) )


def create_results_folder(modelname):
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_folder = os.path.join('/home/s1287/no_backup/s1287/CNN_PET_Prediction_Results/',
                                    str(time_stamp + '_' + modelname + '_multiple/'))
    print(save_folder)
    save_path = Path(save_folder)
    if not save_path.is_dir():
        os.mkdir(save_folder)

    return save_folder


def getCNNModel(config, data):
    if config["model"] == 'Cui_CNN3':
        cnn_model = Model_CUI_CNN3(config=config, data=data)
    elif config["model"] == 'Sharma':
        cnn_model = Model_Sharma(config=config, data=data)
    elif config["model"] == 'Sharma_addConv':
        cnn_model = Model_Sharma_addConv(config=config, data=data)
    elif config["model"] == 'Sharma_addConv2':
        cnn_model = Model_Sharma_addConv2(config=config, data=data)
    return cnn_model