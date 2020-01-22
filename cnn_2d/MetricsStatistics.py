#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:25:36 2018

@author: s1287
"""
import numpy as np
import json

def calculate_metricsStatistics(metrics, save_folder):
    #calculate mean and std of a given metric:
    list_metricnames = ['acc', 'val_acc', 'loss', 'val_loss', 'precision', 'recall']#, 'f1_score']
    metric_stats = []
    
    for item in list_metricnames:
        item_list = [metrics[index][item][0] for index in range(0,len(metrics))]
        
        #calculate mean and std
        if isinstance(item_list[0], str):
            item_list = [float(elem) for elem in item_list]
        
        mean = np.mean(np.asarray(item_list))
        std = np.std(np.asarray(item_list))
        
        print('Mean of ', str(item), ':', str(mean))
        print('Std of ', str(item), ':', str(std))
        
        #save in a specified format for later evaluation
        metric_dict = {'name': item, 'mean': mean, 'std': std}    
        metric_stats.append(metric_dict)
        
    #save the metrics:
    metricStats_filename = save_folder + '/metrics_stats.txt'
    with open(metricStats_filename, 'w') as f:
        f.write(json.dumps(metric_stats))
    print('Metric statistics saved.')
        
    return metric_stats
