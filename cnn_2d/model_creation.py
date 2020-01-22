from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Convolution2D, GlobalAveragePooling2D
import keras.initializers
from keras.regularizers import l2
from keras.layers import LeakyReLU

class Model_FirstCNN:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.config["input_shape2"]))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))


class Model_CUI_CNN3:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Cui et al.
        #input-layer is not added as a layer!
        self.model.add(Conv2D(40, (5, 5), padding='same', activation='relu', 
                              activity_regularizer=l2(0.01), input_shape=self.config["input_shape"]))
        self.model.add(Conv2D(160, (5, 5), padding='same', activation='relu'))
        self.model.add(Conv2D(500, (4, 4), activation='relu'))
        self.model.add(Conv2D(19, (2, 2), activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense(304, activation='relu'))
        #self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')

class Model_CUI_CNN:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Cui et al.
        #input-layer is not added as a layer!
        kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        
        self.model.add(Conv2D(48, (5, 5), padding='same', activation='relu', input_shape=self.config["input_shape"], kernel_initializer=kernel_initializer))
        self.model.add(MaxPooling2D(2,2))
        self.model.add(Conv2D(96, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_initializer))
        self.model.add(Conv2D(700, (4, 4), padding='same', activation='relu', kernel_initializer=kernel_initializer))
        self.model.add(Conv2D(19, (2, 2), activation='relu', kernel_initializer=kernel_initializer))

        self.model.add(Flatten())
        self.model.add(Dense(304, activation='relu'))
       # self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')

class Model_CUI_CNN3_easy:
    def __init__(self, config, data):
        #super(Model_CUI_CNN3, self).__init__()
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Cui et al.
        #input-layer is not added as a layer!
        self.model.add(Conv2D(40, (5, 5), padding='same', activation='relu', input_shape=self.config["input_shape"]))
        #self.model.add(Conv2D(160, (5, 5), padding='same', activation='relu'))
        #self.model.add(Conv2D(500, (4, 4), activation='relu'))
        #self.model.add(Conv2D(19, (2, 2), activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense(304, activation='relu'))
        #self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')
        
class Model_Truong:
    def __init__(self, config, data):
        #super(Model_CUI_CNN3, self).__init__()
        self.config = config
        self.data = data
        self.model = Sequential()
        
        self.build_model()
        
    def build_model(self):
        #nach Truong et al.
        #input-layer is not added as a layer!
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=self.config["input_shape"]))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))   
#        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
#        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        
#        self.model.add(GlobalAveragePooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(304, activation='relu'))
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')

class Model_NN:
    def __init__(self, config, data):
        self.config = config
        self.data = data

        self.model = Sequential()

        self.build_model()

    def build_model(self):

        self.model.add(Dense(200, activation='relu',input_shape= self.config["input_shape"]))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.data.nClasses, activation='softmax'))
        print('Model created successfully.')

class Model_Sharma:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Cui et al.
        #input-layer is not added as a layer!
        self.model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=self.config["input_shape"]))
        self.model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.01)))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.01)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.01)))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.01)))
        
        self.model.add(Flatten())
        self.model.add(Dense((8*8*128), activation='relu'))#, activity_regularizer=l2(0.01)))
        #self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

class Model_Sharma_addConv:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #alpha=0.9
        #input-layer is not added as a layer!
        self.model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=self.config["input_shape"]))
       # self.model.add(LeakyReLU(alpha=alpha))
        self.model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(264, (3,3), padding='same', activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense((8*8*128), activation='relu'))#, activity_regularizer=l2(0.001)))

        #self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')

class Model_Sharma_addConv2:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.model = Sequential()

        self.build_model()

    def build_model(self):
        #nach Cui et al.
        #input-layer is not added as a layer!
        self.model.add(Conv2D(8, (3, 3), padding='same', activation='relu', 
                              #activity_regularizer=l2(0.001), 
                              input_shape=self.config["input_shape"]))
        self.model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))#, activity_regularizer=l2(0.001)))
        self.model.add(Conv2D(264, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        
        self.model.add(Flatten())
        self.model.add(Dense((8*8*128), activation='relu'))#, activity_regularizer=l2(0.001)))
        #self.model.add(Dropout(0.5))
        #classification in 2 classes
        self.model.add(Dense(self.data.nClasses, activation='softmax'))

        print('Model created successfully.')