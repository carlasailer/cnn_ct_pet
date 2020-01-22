from keras.models import Sequential
from keras.layers import Dense, Conv3D, Dropout, Flatten, ZeroPadding3D
import keras.initializers
from keras.regularizers import l2
#from keras.layers import LeakyReLU

class Model_Li1:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Li et al.
        
        #start with a zero-padding layer to account for different padding in diff. axis
        self.model.add(ZeroPadding3D(padding=(3, 0, 0),  input_shape=self.config["input_shape"]))
        self.model.add(Conv3D(10, (4, 5, 5), padding='valid', activation='relu'))
        self.model.add(ZeroPadding3D(padding=(2, 0, 0)))
        self.model.add(Conv3D(10, (3, 5, 5), padding='valid', activation='relu'))
        
        #fully-connected layer to combine 10 feature maps to one feature map
        self.model.add(Conv3D(1, (1, 1 ,1), padding='same', activation='relu'))
        
        #create one fully connected layer to match the output shape
    #    self.model.add(Dense())

        print('Model created successfully.')