import os
import sys
from mdn import *
from collections import deque
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, LSTM, Reshape, ZeroPadding2D, Input, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, AveragePooling2D)
from keras.regularizers import l2

from keras.applications.vgg16 import VGG16
from keras.models import Model


class ResearchModels():
    def __init__(self, model, seq_length, saved_model, imType, transfer_learning = None):

        # set defaults
        self.seq_length  = seq_length
        self.load_model  = load_model
        self.saved_model = saved_model

        # input size
        if imType == 'ON' or imType == 'OFF' or imType == None:
            self.input_shape = (128, 128, self.seq_length) # 4D tensor
        else:
            self.input_shape = (128, 128, 2*self.seq_length) # 4D tensor

        # save the current model
        if model == 'FlowNetSimple':
            self.model = self.FlowNetSimple()
        elif model == 'convLSTM':
            self.model = self.convLSTM()
        elif model == 'VGG_16':
            self.model = self.VGG_16()
        elif model == 'FlowNetSimpleRNN':
            self.model = self.FlowNetSimpleRNN()
        else:
            print("Unknown network.")
            sys.exit()

        # save model to json file
        path = './data/models'
        if not os.path.exists(path):
            os.makedirs(path)
        model_json = self.model.to_json()
        with open(path + "/" + model + ".json", "w") as json_file:
            json_file.write(model_json)

        # load the model (if needed)
        if self.saved_model is not None:
            
            # load json model
            json_file = open('./data/models/' + model + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json, custom_objects={"MixtureDensity": MixtureDensity, "loss":mdn_loss})

            # load weights into new model
            self.model.load_weights(self.saved_model)

        # transfer learning from pre-trained network
        if transfer_learning is not None:
            self.model.load_weights(transfer_learning, by_name = True)
            
        # compile the network
        optimizer = Adam(lr=1e-4)
        self.model.compile(loss=self.custom_MSE, optimizer=optimizer)


    def custom_MSE(self, y_true, y_pred):
        v_pred        = K.reshape(y_pred[:,:-1], [-1, 2])
        ventral_loss  = K.mean(K.square(v_pred - y_true), axis=-1)
        return ventral_loss


    def FlowNetSimple(self):
        mult = 1.0
        reg  = 0.0
        model = Sequential()
        model.add(Conv2D(int(64*mult), (7,7), activation='relu', strides = 2, padding = 'same', input_shape=self.input_shape, kernel_regularizer=l2(reg)))
        model.add(Conv2D(int(128*mult), (5,5), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Conv2D(int(256*mult), (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Conv2D(int(512*mult), (3,3), activation='relu', strides = 1, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Conv2D(int(512*mult), (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Conv2D(int(1024*mult), (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Flatten())
        model.add(Dense(2))
        model.summary()
        return model

    def FlowNetSimpleRNN(self):

        mult = 1.
        reg  = 0.

        # convolution
        image    = Input(shape=self.input_shape, name='image')
        conv2d_1 = Conv2D(int(64*mult), (7,7), activation='relu', strides = 2, kernel_regularizer=l2(reg))(image)
        conv2d_2 = Conv2D(int(128*mult), (5,5), activation='relu', strides = 2, kernel_regularizer=l2(reg))(conv2d_1)
        conv2d_3 = Conv2D(int(256*mult), (3,3), activation='relu', strides = 2, kernel_regularizer=l2(reg))(conv2d_2)
        conv2d_4 = Conv2D(int(512*mult), (3,3), activation='relu', strides = 1, kernel_regularizer=l2(reg))(conv2d_3)
        conv2d_5 = Conv2D(int(512*mult), (3,3), activation='relu', strides = 1, kernel_regularizer=l2(reg))(conv2d_4)
        conv2d_6 = Conv2D(int(1024*mult), (3,3), activation='relu', strides = 1, kernel_regularizer=l2(reg))(conv2d_5)
        flatten  = Flatten()(conv2d_6)
        dense_1  = Dense(128)(flatten)

        exposure     = Input(shape=(1,), name='exposure')
        merged_dense = concatenate([dense_1, exposure])
        dense_2      = Dense(3)(merged_dense)
        model        = Model(inputs=[image, exposure], outputs = [dense_2])
        model.summary()

        return model

    def convLSTM(self):

        reg  = 0.0
        model = Sequential()
        model.add(Conv2D(64, (7,7), activation='relu', strides = 2, padding = 'same', input_shape=self.input_shape, kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Conv2D(128, (5,5), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Conv2D(256, (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Conv2D(512, (3,3), activation='relu', strides = 1, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Conv2D(512, (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Conv2D(1024, (3,3), activation='relu', strides = 2, padding = 'same', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(Reshape((64, 64)))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=False))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(reg)))
        model.add(Dense(2))
        model.summary()
        return model

    def VGG_16(self):

        input_tensor = Input(shape=self.input_shape)
        model = VGG16(weights="imagenet", include_top=False, input_tensor=input_tensor)

        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(2))   

        new_model = Sequential()
        for l in model.layers:
            new_model.add(l)
        for l in new_model.layers:
            l.trainable = False

        new_model.add(top_model)
        new_model.summary()
        return new_model