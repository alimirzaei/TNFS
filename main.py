from keras.datasets import mnist
from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
from utils import getSyntheticDataset
import os
from AEFS_final import AEFS

if __name__ == '__main__':

    
    alpha = 0.001
    beta = 0.01 


    dataset = 'synth_linear_small'

    if(dataset == 'face'):
        data = loadmat('/home/ali/data/fs/warpPIE10P.mat')
        X = data['X']/255.
        x_train = X[0:180,]
        x_test = X[180:209,]
        Y = data['Y']-1
        input_shape = (44,55)
        input_dim = 44*55
        hidden = 128
        alpha = 0.001

    elif(dataset == 'mnist'):
        (x_train, y_train),(x_test,y_test) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), -1)
        x_train = x_train / 255.
        x_test = x_test.reshape(len(x_test), -1)
        x_test = x_test / 255.
        input_shape = (28,28)
        input_dim = 28*28
        hidden = 64
        alpha = 0.01
    
    elif(dataset == 'synth_linear_small'):
        x_train = getSyntheticDataset(N=50000, type='linear')
        x_train = x_train/np.max(x_train)
        hidden = 4
        input_shape = (5,5)
        alpha = 0.01
        beta = 0.01 
    elif(dataset == 'synth_nonlinear_small'):
        x_train = getSyntheticDataset(N=50000, type='nonlinear')
        x_train = x_train/np.max(x_train)
        hidden = 5
        input_shape = (5,5)
        alpha = 0.001
        beta = 0.01 
    elif(dataset == 'synth_linear_large'):
        x_train = getSyntheticDataset(N=50000, indep=5, dep=49, type='linear')
        x_train = x_train/np.max(x_train)
        hidden = 5
        input_shape = (5,50)
        alpha = 0.001
        beta = 0.01 
    elif(dataset == 'synth_nonlinear_large'):
        x_train = getSyntheticDataset(N=50000,indep=5, dep=49, type='nonlinear')
        x_train = x_train/np.max(x_train)
        hidden = 5
        input_shape = (5,50)
        alpha = 0.0005
        beta = 0.01 



    
    model = AEFS(input_dim=x_train.shape[1], encoding_dim=hidden, alpha= alpha, beta=beta)

    import os
    directory = 'AEFS/'+dataset
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(50):
        result = model.train(x_train, batch_size=16, epochs = 5)
        weights = model.getFeatureWeights()
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(131)
        ax.imshow(weights.reshape(input_shape))
        ax = fig.add_subplot(132)
        ax.imshow(x_train[0].reshape(input_shape))
        ax = fig.add_subplot(133)
        y = model.autoencoder.predict(x_train[0].reshape((1,-1)))
        ax.imshow(y.reshape(input_shape))
        fig.savefig('%s/%d.jpg'%(directory,i))
        np.savetxt('%s/val_loss_%d.txt'%(directory,i), result.history['val_loss'])
        np.savetxt('%s/loss_%d.txt'%(directory,i), result.history['loss'])

