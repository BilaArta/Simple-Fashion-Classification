import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD

def CreateModel(images, labels, lr=0.001): # learing rate 0,001
    inputs = Input(shape=(784,))                        # input layer
    h_layer = Dense(128, activation='relu')(inputs)     # node ke 1 / hidden layer ke-1 dengan 128 neuron, dengan fungsi aktivasi relu
    h_layer = Dense(64, activation='relu')(h_layer)     # node ke 2 / hidden layer ke-2 dengan 64 neuron, dengan fungsi aktivasi relu
    outputs = Dense(10, activation='softmax')(h_layer)  # output layer dengan 10 label, dengan fungsi aktivasi softmax
    model = Model(inputs=inputs, outputs=outputs)       # 

    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(images, labels, batch_size=20, epochs=1000, verbose=1) #epoch 1000
    model.save_weights('mlp_weight.h5') #menyimpan hasil pemodelan ke file "mlp_weight.h5"
    return model

def LoadModel(path='mlp_weight.h5'):
    inputs = Input(shape=(784,))                        # input layer
    h_layer = Dense(128, activation='relu')(inputs)     # node ke 1 / hidden layer ke-1 dengan 128 neuron, dengan fungsi aktivasi relu
    h_layer = Dense(64, activation='relu')(h_layer)     # node ke 2 / hidden layer ke-2 dengan 64 neuron , dengan fungsi aktivasi relu
    outputs = Dense(10, activation='softmax')(h_layer)  # output layer dengan 10 label , dengan fungsi aktivasi softmax
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)
    return model

def Predict(images, model):
    predict = model.predict(images)
    return predict
