from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
from pathlib import Path
import numpy as np

from scripts_CNN.loss_ssim import ssim_fct, calc_ssim_git
from scripts_CNN.ssim_callback import MetricsCallback
       
        
class Trainer:
    def __init__(self, config, model, data, save_folder):
        self.config = config
        self.training_model = model
        self.data = data
        self.save_folder = save_folder
        self.optimizer = None
        
        self.train()

    def train(self):
        #create new (sub)folder for saving
    #    self.save_folder = self.folder + '_datasetnr_'+ str(self.data.dataset_nr)
        print(self.save_folder)
        save_path = Path(self.save_folder)
        if not save_path.is_dir():
            os.mkdir(self.save_folder)

        #settings from config (main.py)
        batch_size = self.config["batch_size"]
        #loss = self.config["loss_function"]
        self.epochs = self.config["epochs"]

        #get optimizer with specific settings
        self.optimizer = self.getOptimizer()

        #use custom metric to be calculated after each epoch
        self.metrics_callback = MetricsCallback(validation_data=(self.data.test_data, self.data.test_labels))
           
        loss = self.config["loss_function"]#mean_squared_error
        
        #compile the model with the given optimizer and the loss function
        self.training_model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])

        self.training_model.summary()

        # train the model
        self.history = self.training_model.fit(x=self.data.train_data, y=self.data.train_labels,
                                               batch_size=batch_size, epochs=self.epochs, verbose=1,
                                               validation_data=(self.data.test_data, self.data.test_labels),
                                               callbacks=[self.metrics_callback])
        #metrics.get_data()

        print('Model trained successfully.')

    def getOptimizer(self):
        self.learning_rate = self.config["learning_rate"]

        if self.config["optimizer"] == 'sgd':
            decay = 1e-6
            momentum = 0.9
            optimizer = optimizers.SGD(lr=self.learning_rate, decay=decay, momentum=momentum, nesterov=True)

        elif self.config["optimizer"] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)

        elif self.config["optimizer"] == 'Adagrad':
            optimizer = optimizers.Adagrad(lr=self.learning_rate, epsilon=None, decay=0.0)

        elif self.config["optimizer"] == 'Adam':
            optimizer = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                        decay=0.0, amsgrad=False)
        return optimizer
    
#    def start_DataAugmentation(self):
#        current_augmentation = Data_Augmentation.Data_Augmentation()
#        #create the augmented data 
#        current_augmentation.dataset_gen.fit(self.data.train_data)
#        self.training_model.fit_generator(current_augmentation.dataset_gen.flow(self.data.train_data, 
#                                                                                self.data.train_labels,
#                                                                                batch_size=32),
#                steps_per_epoch=len(self.data.train_data)/32, epochs=self.epochs)