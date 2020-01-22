from keras.models import Sequential
from model_creation import Model_FirstCNN
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import os
from pathlib import Path

import Data_Augmentation
 
class Trainer:
    def __init__(self, config, model, data, save_folder):
        self.config = config
        self.training_model = model
        self.data = data
        self.folder = save_folder
        self.optimizer = None
        
        self.train()

    def train(self):
        #create new (sub)folder for saving
        self.save_folder = self.folder + 'datasetnr_'+ str(self.data.dataset_nr)
        print(self.save_folder)
        save_path = Path(self.save_folder)
        if not save_path.is_dir():
            os.mkdir(self.save_folder)

        #settings from config (main.py)
        batch_size = self.config["batch_size"]
        loss = self.config["loss_function"]
        self.epochs = self.config["epochs"]

        #get optimizer with specific settings
        self.optimizer = self.getOptimizer()

        metrics = ['accuracy']

        self.training_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

        self.training_model.summary()
        
        #data augmentation
        self.start_DataAugmentation()

        callback_earlyStopping = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0,
                                                mode='auto'),
        #callback_bestModel =
                                  ModelCheckpoint(filepath=str(self.save_folder + '/model-{epoch:03d}.h5'),
                                             verbose=1,
                                             monitor='val_loss', save_best_only=True,
                                             save_weights_only=False, mode='auto', period=1)
                                  ]

        #use class weights: use the opposite weight as in the distribution
        if self.config["class_weights"] == True:
            class_weight_low = self.data.n_highsuv_train/self.data.n_lowsuv_train
            class_weight_high = 1
            class_weight = {0: class_weight_low, 1: class_weight_high}
            print(class_weight)

            #train the model
            self.history = self.training_model.fit(x=self.data.train_data, y=self.data.train_labels,
                                                   callbacks=callback_earlyStopping,
#                                                   callbacks=callback_bestModel,
                                                   batch_size=batch_size, epochs=self.epochs, verbose=1,
                                                   class_weight=class_weight,
                                                   validation_data=(self.data.test_data, self.data.test_labels),)
        else:
            # train the model
            self.history = self.training_model.fit(x=self.data.train_data, y=self.data.train_labels,
                                                   batch_size=batch_size, epochs=self.epochs, verbose=1,
#                                                   callbacks=callback_earlyStopping,
                                                   validation_data=(self.data.test_data, self.data.test_labels))

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
    
    def start_DataAugmentation(self):
        current_augmentation = Data_Augmentation.Data_Augmentation()
        #create the augmented data 
        current_augmentation.dataset_gen.fit(self.data.train_data)
        self.training_model.fit_generator(current_augmentation.dataset_gen.flow(self.data.train_data, 
                                                                                self.data.train_labels,
                                                                                batch_size=32),
                steps_per_epoch=len(self.data.train_data)/32, epochs=self.epochs)