from scipy.io import loadmat
import numpy as np

def load_data(dir ='/home/ali/Datasets/Channel', SNR = 22):
    noisy = loadmat(dir + '/My_noisy_H_%d.mat'%SNR)['My_noisy_H']
    noisy_image = np.zeros((40000,72,14,1))
    noisy_image[:,:,:,0] = np.real(noisy)
    perfect = loadmat(dir + '/My_perfect_H_%d.mat'%SNR)['My_perfect_H']
    perfect_image = np.zeros((40000,72,14,1))
    perfect_image[:,:,:,0] = np.real(perfect)

    return (noisy_image, perfect_image)

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def getSyntheticDataset(N = 10000,type = 'linear', indep = 5, dep = 4):
    X = np.zeros((N, indep + indep*dep))
    index = 0
    for i in range(indep):
        X[:, index] = np.random.rand(N)
        index = index + 1
        for j in range(dep):
            if(type == 'linear'):
                X[:, index+j] = (j+2)*X[:, index-1] 
            else:
                X[:, index+j] = sigmoid(X[:, index-1]*(3*float(j)/dep+1))

        index = index + dep
    return X

if __name__ == '__main__':

    X = getSyntheticDataset()

    print(X[0,:])


