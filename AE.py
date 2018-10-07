from utils import getSyntheticDataset, load_data
from keras.models import Model, load_model
from keras.layers import Dense,Input
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from keras.datasets import mnist

epochs_ae = 10
epochs_fs = 10

dataset = 'channel' # face , mnist, channel

if(dataset == 'face'):
    data = loadmat('/home/ali/Datasets/fs/warpPIE10P.mat')
    X = data['X']/255.
    Y = data['Y']-1
    shape = (44,55)
    input_dim = 44*55
    hidden = 20
    l1 = 0.01

elif(dataset == 'mnist'):
    (x_train, y_train),(_,_) = mnist.load_data()
    X = x_train.reshape(-1, 28*28)
    X = X / 255.
    #X1= X2 = X+.000001
    shape = (28,28)
    epochs_ae = 5
    epochs_fs = 5
    #input_dim = 28*28
    hidden = 20
    #thresh = .5
    l1=.01
    #l2=0
elif(dataset == 'channel'): 
    X1, X2 = load_data()
    #X = X1.reshape(len(X1),-1)
    X = X2.reshape(len(X2),-1)
    shape = (72, 14)
    input_dim = 72*14
    hidden = 100
    l1 = .001
    l2 = 0
elif(dataset == 'synth_linear_small'):
    X = getSyntheticDataset(N=10000, indep=5 , dep=4, type='linear')
    hidden = 5
    shape = (5, 5)
    l1= 0.01 # linear 0.1 is good
    epochs_fs= 10
elif(dataset == 'synth_nonlinear_small'):
    X1 = getSyntheticDataset(N=10000, indep=5 , dep=4, type='nonlinear')
    X1 = X1 / np.max(X1)
    X2 = X1
    hidden = 5
    thresh = .5
    shape = (5, 5)
    input_dim = 25
    l1= 0.12 # linear 0.1 is good
    l2 = 0
elif(dataset == 'synth_linear_large'):
    X = getSyntheticDataset(N=10000, indep=5 , dep=49, type='linear')
    X = X / np.max(X)
    hidden = 5
    shape = (5, 50)
    input_dim = 5*50
    l1= 0.005 # linear 0.1 is good
    l2 = 0
elif(dataset == 'synth_nonlinear_large'):
    X1 = getSyntheticDataset(N=10000, indep=5 , dep=49, type='nonlinear')
    X1 = X1 / np.max(X1)
    X2 = X1
    hidden = 5
    thresh = .5
    shape = (5, 50)
    input_dim = 250
    l1= 0.8 # linear 0.1 is good
    l2 = 0

# if(dataset =='synth'):
#     l1 = .01
#     X = getSyntheticDataset(10000)
# elif(dataset == 'mnist'):

dim = X.shape[1]

x1 = Input((dim,))
x2 = Dense(hidden, activation='sigmoid')(x1)
x3 = Dense(dim)(x2)

autoencoder = Model(x1, x3)
encoder = Model(x1, x2)
autoencoder.compile(optimizer='adam', loss='mse')


try:
    autoencoder.load_weights('ae.pkl')
except:
    autoencoder.fit(X,X, epochs=epochs_ae)
    autoencoder.save_weights('ae.pkl')


L = encoder.predict(X)

def layer1_reg(weight_matrix):
        return l1*K.sum(K.sqrt(K.tf.reduce_sum(K.square(weight_matrix), axis=1)))

y1 = Input((dim,))
y2 = Dense(2*hidden, activation='sigmoid', kernel_regularizer =layer1_reg)(y1)
y3 = Dense(hidden, activation='sigmoid')(y2)


def loss_mse(y, y_true):
    return K.tf.reduce_mean(K.tf.square(y-y_true))
model = Model(y1, y3)
model.compile(optimizer='adam', loss='mse', metrics=[loss_mse])
try:    
    model.load_weights('model.pkl')
except:
    model.fit(X, L, epochs=epochs_fs)
    model.save_weights('model.pkl')

w = model.layers[1].get_weights()[0]
w = np.sum(np.square(w),1)
print(w)
plt.figure()
plt.imshow(w.reshape(shape))
plt.figure()
plt.imshow(X[0].reshape(shape))
plt.figure()
w2 = np.copy(w)
w2[w<np.sort(w)[-20]] = 0
w2[w>=np.sort(w)[-20]] = 1
plt.imshow(w2.reshape(shape))
plt.show()
