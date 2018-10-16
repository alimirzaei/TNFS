import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from RRFS import RRFS
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical   
from keras.callbacks import EarlyStopping
import pickle
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
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

def test_softmax(x_train, y_train, x_test, y_test, features):
    early_stopping = EarlyStopping(patience=2)
    x_test = x_test.reshape(len(x_test), -1)
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test[:,features]
    x_train = x_train[:,features]
    num_classes = len(np.unique(y_train))
    model = Sequential([Dense(num_classes, input_dim = len(features), activation='softmax')])
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    y_binary_train = to_categorical(y_train, num_classes=num_classes)
    y_binary_test = to_categorical(y_test, num_classes=num_classes)
    model.fit(x_train, y_binary_train, epochs=1000, validation_split=.15, verbose=0)
    result = model.evaluate(x_test, y_binary_test)
    return result[1]



if __name__ == '__main__':

    datasets = ['mnist.mat','COIL20.mat','Yale.mat',
    'PCMAC.mat','BASEHOCK.mat','RELATHE.mat','Prostate_GE.mat' ,'Isolet.mat']
    #datasets = ['RELATHE.mat']
    info = {'Isolet.mat' : {'test' : 780, 'train':780},
            'Prostate_GE.mat' :{'test' : 50, 'train':50},
            'mnist.mat':{'test':27100,'train':1000},
            'Yale.mat':{'test':75,'train':95},
            'COIL20.mat':{'test':720, 'train':720},
            'PCMAC.mat':{'test':960,'train':960},
            'RELATHE.mat':{'test':648,'train':648},
            'BASEHOCK.mat':{'test':994, 'train':994}}

    for dataset in datasets:
        data = loadmat('/home/ali/Datasets/fs/'+dataset)
        
        X = data['X']
        X = X.astype(float)/255.
        dim = X.shape[1]
        Y = data['Y']-1
        Y = Y.reshape((len(Y),))
        classes = np.unique(Y)
        num_classes = len(classes)
        test_perclass_number = int(info[dataset]['test']/num_classes) 
        train_perclass_number = int(info[dataset]['train']/num_classes)

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

        #scaler = preprocessing.MinMaxScaler()        
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)
        
        mmax = np.max(x_train)
        mmin = np.min(x_train)

        x_train = (x_train-mmin)/(mmax-mmin)
        x_test = (x_test-mmin)/(mmax-mmin)
        rrfs = RRFS(dim, loss='mse')
        
        rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
        
        ps = [2, 4, 6, 8, 10, 20 , 30, 40 , 50, 60, 70, 80, 100]
        l1s = [1e-5,0.0001,.001,.005,.01,.05,.1]
        accs_ps = np.zeros(len(ps))
        fatures_ps_l1s = {}
        accs_ps_l1s = {}
        for i, p in enumerate(ps):
            num_features = int(p*dim/100)
            accs_l1 = np.zeros(len(l1s))
            for index,l1 in enumerate(l1s):
                w = rrfs.train_fs_network(x_train, l1=l1, name=dataset+'_fs.hd5', epochs=1000, loss='mse')
                features = np.argsort(w)[-num_features:]
                fatures_ps_l1s[(p,l1)] = features
                #accs_l1[index], std = test_kmeans(x_test, y_test, features, number=20)
                accs_l1[index] = test_softmax(x_train, y_train, x_test, y_test, features)
                accs_ps_l1s[(p,l1)]= accs_l1[index]
                print(accs_l1[index])
            accs_ps[i] = np.max(accs_l1)
            print(dataset)
            print(accs_ps)
            save_dict(accs_ps, 'results/classification/final_accs_%s.npy'%dataset)
            save_dict(fatures_ps_l1s, 'results/classification/features_%s.npy'%dataset)
            save_dict(accs_ps_l1s, 'results/classification/all_accs_%s.npy'%dataset)
        
