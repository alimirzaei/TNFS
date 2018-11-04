from RRFS import RRFS
from sklearn.manifold import Isomap,TSNE,LocallyLinearEmbedding,MDS, SpectralEmbedding
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import numpy as np


def my_isomap(X, y=None, l1 = .1, n_components=2):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = Isomap(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_tsne(X, y=None,l1 = .1, n_components=2):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = TSNE(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_lle(X, y=None,l1 = .1, n_components=2):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = LocallyLinearEmbedding(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_mds(X, y=None,l1 = .1, n_components=2):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = MDS(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx



def my_se(X, y=None,l1 = .1, n_components=2):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = SpectralEmbedding(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def laplacian_score(X, y=None):
    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)
    
    # obtain the scores of features
    score = lap_score.lap_score(X, W=W)

    # sort the feature scores in an ascending order according to the feature scores
    idx = lap_score.feature_ranking(score)

    return idx

def udfs_score(X, y, gamma=.1):
    Weight = UDFS.udfs(X, gamma=gamma, n_clusters=len(np.unique(y)))
    idx = feature_ranking(Weight)
    return idx


