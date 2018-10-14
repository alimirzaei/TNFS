from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import pandas as pd
from RRFS import RRFS

def test_kmeans(x_test, y_test, faetures, number=20):
    x = x_test.reshape(len(x_test), -1)
    x = x[:,faetures]
    n_clusters = len(set(y_test))
    res = np.zeros(number)
    for index in range(number):
        km = KMeans(n_clusters=n_clusters, n_jobs=4, init='random')
        y = km.fit_predict(x)
        cost = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                cost[i,j] = -sum(np.logical_and(y==i,y_test==j))
        row_ind, col_ind = linear_sum_assignment(cost)
        y_changed = 100*np.ones_like(y)
        for i,j in zip(row_ind, col_ind):
            y_changed[y==i] = j
        res[index] = float(np.sum(y_changed==y_test))*100./float(len(y))
    return (np.mean(res), np.std(res))

def test_knn(x_train, y_train, x_test, y_test, features):
    x_test = x_test.reshape(len(x_test), -1)
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test[:,features]
    x_train = x_train[:,features]
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
    knn.fit(x_train, y_train)
    y = knn.predict(x_test)
    return np.sum(y==y_test)*100/len(y)

if __name__ == '__main__':

    datasets = ['CLL_SUB_111.mat','madelon.mat','TOX_171.mat',
    'warpPIE10P.mat','Isolet.mat','PCMAC.mat','USPS.mat',
    'lung_discrete.mat','Prostate_GE.mat' ,'warpAR10P.mat','mnist.mat']
    datasets = ['mnist.mat']
    info = {'Isolet.mat' : {'test' : 780, 'train':780},
            'Prostate_GE.mat' :{'test' : 50, 'train':50},
            'mnist.mat':{'test':27100,'train':1000}}

    for dataset in datasets:
        data = loadmat('/home/ali/Datasets/fs/'+dataset)
        
        X = data['X']
        X = X.astype(float)/255.
        dim = X.shape[1]
        Y = data['Y']-1
        Y = Y.reshape((len(Y),))
        classes = np.unique(Y)
        num_classes = len(classes)
        test_perclass_number = info[dataset]['test']/num_classes 
        train_perclass_number = info[dataset]['train']/num_classes

        test_indexes = []
        train_indexes = []

        for c in classes:
            c_indexes = np.where(Y==c)[0]
            c_train_indexes = np.random.choice(c_indexes, train_perclass_number, False)
            c_test_indexes = np.random.choice(list(set(c_indexes)-set(c_train_indexes)), test_perclass_number, False)
            train_indexes += list(c_train_indexes)
            test_indexes += list(c_test_indexes)

        x_test = X[test_indexes,:]
        y_test = Y[test_indexes]

        x_train = X[train_indexes,:]
        y_train = Y[train_indexes]

        #scaler = preprocessing.StandardScaler()        
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)
        rrfs = RRFS(dim, loss='mse')
        rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=100)
        
        ps = [2, 4, 8, 10, 20 , 30, 40 , 50]
        l1s = [0.0001,.001,.005,.01,.05,.1]
        accs_ps = np.zeros(len(ps))
        for i, p in enumerate(ps):
            num_features = int(p*dim/100)
            accs_l1 = np.zeros(len(l1s))
            for index,l1 in enumerate(l1s):
                w = rrfs.train_fs_network(x_train, l1=l1, name=dataset+'_fs.hd5', epochs=100, loss='mse')
                features = np.argsort(w)[-num_features:]
                accs_l1[index], std = test_kmeans(x_test, y_test, features, number=20)
                print(accs_l1[index])
            accs_ps[i] = np.max(accs_l1)
            print(accs_ps)

 