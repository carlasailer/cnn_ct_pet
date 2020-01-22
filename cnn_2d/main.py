import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os, sys
import gc
import matplotlib.pyplot as plt

from main_single import run_main_single
from main_multiple import run_main_multiple

# change matplotlib backend to avoid error later
plt.switch_backend('Agg')

# disable pycache folder:
sys.dont_write_bytecode = False

## handle GPU ressources
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
current_session = tf.Session(config=config)
set_session(current_session)

gc.collect()

###### settings 
epochs = 100
input_shape = (8, 8, 1)
path = '/home/s1287/med_data/PET_prediction/datasets/old/'

#parameters to be changed:
dataset_type = 'single' #'single', 'multiple'
loss = 'categorical_crossentropy'
learning_rate = 0.01
batch_size = 64
optimizer = 'Adam'
subtract_mean = True
#model = 'Cui_CNN3'
model = 'Sharma_addConv'

#fixed parameters:
apply_class_weights = False
small_dataset = True
scaling = True
if dataset_type == 'single':
    dataset_nr = 0
elif dataset_type == 'multiple':
    dataset_nr = [0,3,4,7,8,9,10,13,19]#18,19]


config = {"epochs":             epochs,
        "loss_function":        loss,
        "learning_rate":        learning_rate,
        "batch_size":           batch_size,
        "input_shape":          input_shape,
        "optimizer":            optimizer,
        "input_path":           path,
        "dataset_nr":           dataset_nr,
        "mean_subtrac":         subtract_mean,
        "scaling":              scaling,
        "class_weights" :       apply_class_weights,
        "small_dataset":        small_dataset,
        "model":                model
          }

#run the chosen main method including model creation, training, evaluation and saving
if dataset_type == 'single':
    run_main_single(config)
elif dataset_type == 'multiple':
    run_main_multiple(config)

# end the tf session to release GPU ressources
current_session.close()
    