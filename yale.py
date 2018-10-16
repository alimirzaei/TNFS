from experiments_fs import test_kmeans
from scipy.io import loadmat
import numpy as np
from RRFS import RRFS
import matplotlib.pyplot as plt

info = {'Yale.mat' : {'test' : 75, 'train':90}}

dataset = 'Yale.mat'
data = loadmat('/home/ali/Datasets/fs/'+dataset)

X = data['X']

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

t_max = np.max(x_train)
t_min = np.min(x_train)

#scaler = preprocessing.StandardScaler()        
x_train = (x_train.astype(float)-t_min)/(t_max-t_min)
x_test = (x_test.astype(float)-t_min)/(t_max-t_min)
#x_test = scaler.transform(x_test)
rrfs = RRFS(dim, loss='mse')
rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)

ps = [2, 4, 6, 8, 10, 20 , 30, 40 , 50, 60, 70, 80, 100]
l1s = [0.0001,.001,.005,.01,.1,1]
accs_ps = np.zeros(len(ps))


max_features = []
all_features ={}
for i, p in enumerate(ps):
    num_features = int(p*dim/100)
    print('NUM === %d'%num_features)
    accs_l1 = np.zeros(len(l1s))
    acc_max = 0
    for index,l1 in enumerate(l1s):
        w = rrfs.train_fs_network(x_train, l1=l1, name=dataset+'_fs.hd5', epochs=100, loss='mse')
        features = np.argsort(w)[-num_features:]
        acc, std = test_kmeans(x_test, y_test, features, number=1)
        if(acc>acc_max):
            acc_max = acc
            max_features = features
        print(acc)
    accs_ps[i] = acc_max
    print(accs_ps)
    all_features[p] = max_features

pics = [0,10,20]
fig = plt.figure(figsize=(len(ps),len(pics)))

for index,i in enumerate(pics): 
    x = x_test[i]
    sampled_x = np.zeros(1024)
    for j,p in enumerate(ps):
        sampled_x[all_features[p]] = np.copy(x[all_features[p]])
        ax = fig.add_subplot(len(pics), len(ps),len(ps)*index+ j+1)  
        ax.imshow(sampled_x.reshape((32,32)).T)
        
fig.show()
fig.savefig('yale.jpg')
