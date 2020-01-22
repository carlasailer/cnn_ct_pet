 #%matplotlib inline
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os, sys
import gc
import matplotlib.pyplot as plt

from scripts_CNN.main_allSamples import run_main

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
epochs = 250
#size of CT patch
input_shape = (6, 13, 13, 1)
path = '/home/s1287/no_backup/s1287/results_interp/patches_for_CNN/'

#parameters to be changed:
loss = 'mean_squared_error'
learning_rate = 0.01
batch_size = 32
optimizer = 'sgd'
subtract_mean = True
model = 'Li1'

#fixed parameters:
scaling = True


config = {"epochs":             epochs,
        "loss_function":        loss,
        "learning_rate":        learning_rate,
        "batch_size":           batch_size,
        "input_shape":          input_shape,
        "optimizer":            optimizer,
        "input_path":           path,
        "scaling":              scaling,
        "model":                model
          }

run_main(config)

# end the tf session to release GPU ressources
current_session.close()
    