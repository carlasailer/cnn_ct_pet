import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix
import numpy as np

class Evaluator:
    def __init__(self, config, model, data, history):
        self.config = config
        self.evaluation_model = model
        self.data = data
        self.history = history
        self.metrics = None
        #default values
        self.evaluate()


    def evaluate(self):
        self.evaluation_model.evaluate(self.data.test_data, self.data.test_labels)
                
        y_true = np.array([np.argmax(label, axis=None, out=None) for label in self.data.test_labels])
        y_predict_classes = self.evaluation_model.predict_classes(self.data.test_data)
        y_predict_probs = self.evaluation_model.predict(self.data.test_data)
        print('Class probabilities are: ')
        print(y_predict_probs)

        #metrics 
        #get confusion matrix and split it into components
        confusion_mx = confusion_matrix(y_true, y_predict_classes)
       
        loss =     self.history.history['loss']
        val_loss = self.history.history['val_loss']
        acc =      self.history.history['acc']
        val_acc =  self.history.history['val_acc']
            
        #put into dictionary:
        self.metrics = {
                'loss':             loss,
                'val_loss':         val_loss,
                'acc':              acc,
                'val_acc':          val_acc,
                'confusion_matrix': confusion_mx.tolist()
        }

        #calculate recall and precision
        self.calcRecallAndPrecision(confusion_mx)
        
        #print results to console
        print('Confusion matrix: ',     confusion_mx)
        print(self.metrics)
        print('Model evaluated successfully.')

    def calcRecallAndPrecision(self, conf_mx):
        tn, fp, fn, tp = conf_mx.ravel()

        if tp != 0:
            #calculate and round precision and recall
            precision = np.round((tp / (tp+fp)), 5)
            recall = np.round((tp / (tp+fn)), 5)
            f1_score = 2*(precision*recall / (precision+recall))
            
        else:
            precision = 0
            recall = 0
            f1_score = 0

        #add to the metrics
        self.metrics['precision'] = str(precision)
        self.metrics['recall'] = str(recall)
        self.metrics['f1_score'] = str(f1_score)
        
        #print results
        print('Precision: ', precision)
        print('Recall: ',    recall)
        print('f1_score: ',  f1_score)