
from keras.models import Sequential , Model
from keras.layers import Input, Dense
from keras.regularizers import  l1, l2
import keras.backend as K
from keras import regularizers
from keras.activations import linear
from keras.datasets import mnist
import tensorflow as tf
from keras.losses import mean_squared_error
from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
from utils import getSyntheticDataset





class AEFS():

    def __init__(self, input_dim=44*55, encoding_dim = 128, alpha = 0.01, beta = 0.01):
        self.alpha = alpha
        self.beta = beta 
        self.input_img = Input(shape=(input_dim,))
        self.encoded = Dense(encoding_dim, activation='relu' , kernel_regularizer =self.layer1_reg)(self.input_img)
        self.decoded = Dense(input_dim, activation='sigmoid' , kernel_regularizer =self.layer2_reg)(self.encoded)
        self.autoencoder = Model(self.input_img, self.decoded)
        #opt = tf.train.ProximalGradientDescentOptimizer(0.003)
        self.autoencoder.compile(optimizer= 'Adadelta', loss= 'mse')#self.frob_loss)
        self.autoencoder.summary()

    
    def train(self, X, epochs=150, batch_size=20):
        return self.autoencoder.fit(X, X,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True, validation_split=0.15)
    
    def getFeatureWeights(self):
        weights1 = self.autoencoder.layers[1].get_weights()[0]
        return np.sum(np.square(weights1),1)

    def layer1_reg(self, weight_matrix):
        return self.alpha * K.sum(K.sqrt(tf.reduce_sum(K.square(weight_matrix), axis=1))) + (self.beta/2.)*K.sum(K.square(weight_matrix))

    def layer2_reg(self, weight_matrix):
        return (self.beta/2.)*K.sum(K.square(weight_matrix))

    def frob_loss(self, y_true,y_pred):
        return 0.5*K.mean(K.sqrt(K.sum(tf.reduce_sum(K.square(y_true-y_pred)))))

        


