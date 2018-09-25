
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


alpha = 0.001
beta = 0.01 





class AEFS():

    def __init__(self, input_dim=44*55, encoding_dim = 128, alpha = 0.01, beta = 0.01):
        self.alpha = alpha
        self.beta = beta 
        self.input_img = Input(shape=(input_dim,))
        self.encoded = Dense(encoding_dim, activation='relu' , kernel_regularizer =self.layer1_reg)(self.input_img)
        self.decoded = Dense(input_dim, activation='sigmoid' , kernel_regularizer =self.layer2_reg)(self.encoded)
        self.autoencoder = Model(self.input_img, self.decoded)
        #opt = tf.train.ProximalGradientDescentOptimizer(0.003)
        self.autoencoder.compile(optimizer= 'Adadelta', loss= self.frob_loss)
        self.autoencoder.summary()

    
    def train(self, X, epochs=150, batch_size=20):
        self.autoencoder.fit(X, X,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True, validation_split=0.15)
    
    def getFeatureWeights(self):
        weights1 = self.autoencoder.layers[1].get_weights()[0]
        return np.sum(np.square(weights1),1)

    def layer1_reg(self, weight_matrix):
        return self.alpha * K.sum(K.sqrt(tf.reduce_sum(K.square(weight_matrix), axis=1))) + (beta/2.)*K.sum(K.square(weight_matrix))

    def layer2_reg(self, weight_matrix):
        return (self.beta/2.)*K.sum(K.square(weight_matrix))

    def frob_loss(self, y_true,y_pred):
        return 0.5*K.mean(K.sqrt(K.sum(tf.reduce_sum(K.square(y_true-y_pred)))))

        

if __name__ == '__main__':
    
    X = getSyntheticDataset(N=50000, type='nonlinear')
    X = X/np.max(X)
    model = AEFS(input_dim=X.shape[1], encoding_dim=4)
    model.train(X, batch_size=16, epochs = 10)
    weights = model.getFeatureWeights()
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(131)
    ax.imshow(weights.reshape((5,5)))
    ax = fig.add_subplot(132)
    ax.imshow(X[0].reshape((5,5)))
    ax = fig.add_subplot(133)
    y = model.autoencoder.predict(X[0].reshape((1,25)))
    ax.imshow(y.reshape((5,5)))
    fig.savefig('aefs_nonlinear.jpg')
    





# if __name__ == '__main__':
#     dataset = 'face'
#     if(dataset == 'face'):
#         data = loadmat('/home/ali/data/fs/warpPIE10P.mat')
#         X = data['X']/255.
#         x_train = X[0:180,]
#         x_test = X[180:209,]
#         Y = data['Y']-1
#         input_shape = (44,55)
#         input_dim = 44*55

#     elif(dataset == 'mnist'):
#         (x_train, y_train),(x_test,y_test) = mnist.load_data()
#         x_train = x_train.reshape(len(x_train), -1)
#         x_train = x_train / 255.
#         x_test = x_test.reshape(len(x_test), -1)
#         x_test = x_test / 255.
#         input_shape = (28,28)
#         input_dim = 28*28
    
#     elif()


    
# encoding_dim = 128  






# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)




# n = 10 
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(44, 55))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(44, 55))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


# for layer in autoencoder.layers:
#     weights = layer.get_weights() # list of numpy arrays

# weights1 = weights[0]
# layer1_weights = np.sum(np.square(weights1),0)





