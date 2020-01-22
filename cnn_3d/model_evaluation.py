import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix
import numpy as np
from scripts_CNN.ssim_callback import calc_ssim

class Evaluator:
    def __init__(self, config, model, data, history, save_folder, training_ssim):
        self.config = config
        self.evaluation_model = model
        self.data = data
        self.history = history
        self.save_folder = save_folder
        self.training_ssim = training_ssim
        self.metrics = None
        
        self.evaluate()


    def evaluate(self):
        self.evaluation_model.evaluate(self.data.test_data, self.data.test_labels)
        self.y_predict = self.evaluation_model.predict(self.data.test_data)
       
        loss     = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        ssim     = self.training_ssim
        val_ssim = calc_ssim(y_true=self.data.test_labels, y_pred=self.y_predict)*100
            
        #save to dictionary:
        self.metrics = {
                'loss':             loss,
                'val_loss':         val_loss,
                'ssim':             ssim,
                'val_ssim':         val_ssim
                }

        print(self.metrics)
        print('Model evaluated successfully.')
