from scipy.io import loadmat
from glob import glob
import  matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from keras.layers import Dense
from keras.models import Sequential


methods = glob('results/new/other/*.pkl')

dataset = 'mnist_subset'
number_features = 15
shape= (28, 28)


mat = loadmat('/home/ali/Datasets/fs/%s.mat'%dataset)
X = mat['X']    # data
X = X.astype(float)
y = mat['Y']    # label
y = y[:, 0]
labels = np.unique(y)

# random selection; one image per class
pics = []
for l in labels:
    indexes = np.argwhere(y==l)
    selected = random.choice(indexes[:])
    pics.append(selected)

# check existence of dataset and method
methods_result = []
titles = []
for method in methods:
    with open(method,'rb') as f:
        results = pickle.load(f)
    if dataset not in results.keys():
        continue
    methods_result.append(results[dataset]['feature_ranking'][:number_features])
    titles.append(method.split('/')[-1].split('.')[0])


fig = plt.figure(figsize=(len(titles)+1 ,len(pics)+1))
rows = len(pics)+1
columns = len(titles)+1

#plot original pics
for index, pic in enumerate(pics):
    x = X[pic]
    ax = fig.add_subplot(rows, columns, index*columns+1)
    ax.imshow(x.reshape(shape))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
for mindex,(title,selected_features) in enumerate(zip(titles,methods_result)):
    model = Sequential()
    model.add(Dense(100, input_dim=len(selected_features), activation = 'relu'))
    model.add(Dense(X.shape[1]))
    model.compile(optimizer='Adam', loss='mse')
    model.fit(X[:,selected_features], X, epochs=1000)


    sampled_x = np.zeros(X.shape[1])
    sampled_x[selected_features] = 1
    
    # plot reconstruction images
    for index,i in enumerate(pics):
        x = X[i,:].T
        estimated = model.predict(x[selected_features].reshape(1,-1))
        ax = fig.add_subplot(rows, columns, index*columns+2+mindex)
        ax.imshow(estimated.reshape(shape))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')

    # plot selected features
    ax = fig.add_subplot(rows, columns, len(pics)*columns+2+mindex)
    ax.imshow(sampled_x.reshape(shape))
    ax.set_xlabel(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
fig.show()
plt.show()

    
