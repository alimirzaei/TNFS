from scipy import io
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import numpy as np
import pickle
import os


def evaluate_clustering(selected_features,y):
    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = np.zeros(20)
    acc_total = np.zeros(20)
    for i in range(0, 20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=len(np.unique(y)), y=y)
        nmi_total[i]= nmi
        acc_total[i]= acc

    # output the average NMI and average ACC
    return (np.mean(nmi_total), np.std(nmi_total)), (np.mean(acc_total),np.std(acc_total))

def evaluate_classification(selected_features, y):
    clf = MLPClassifier()
    scores = cross_validate(clf, selected_features, y, cv=5, n_jobs=4)
    return (np.mean(scores['test_score']), np.std(scores['test_score']))

def main():
    directory = 'results'    
    if not os.path.exists(directory):
        os.makedirs(directory)
    method = 'LapScore'
    result_file_path = '%s/%s.pkl'%(directory, method)
    if(os.path.exists(result_file_path)):
        with open(result_file_path, 'rb') as f:
            results = pickle.load(f)    
    else:
        results={}
    datasets = ['COIL20','Yale','PCMAC','BASEHOCK','RELATHE','Prostate_GE' ,'Isolet', 'mnist']
    for dataset in datasets:
        if(dataset in results.keys()):
            print('READ RESULTS DATASET %s SUCCESSFULLY'%dataset)
            continue
        results[dataset] = {}
        # load data
        mat = io.loadmat('/home/ali/Datasets/fs/%s.mat'%dataset)
        X = mat['X']    # data
        X = X.astype(float)
        y = mat['Y']    # label
        y = y[:, 0]

        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        
        # obtain the scores of features
        score = lap_score.lap_score(X, W=W)

        # sort the feature scores in an ascending order according to the feature scores
        idx = lap_score.feature_ranking(score)

        # perform evaluation on clustering task
        
        percents = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 100]
        
        
        
        results[dataset]['mean'] = np.zeros((3, len(percents)))
        results[dataset]['std'] = np.zeros((3, len(percents)))
        for index, p in enumerate(percents):
            # obtain the dataset on the selected features
            num_fea = int(p*X.shape[1]/100)    # number of selected features
            selected_features = X[:, idx[0:num_fea]]

            (classification_accuracy_mean, classification_accuracy_std) = evaluate_classification(selected_features,y)
            (clustering_nmi_mean, clustering_nmi_std), (clustering_accuracy_mean, clustering_accuracy_std) = evaluate_clustering(selected_features, y)
            
            
            results[dataset]['mean'][:,index] = [classification_accuracy_mean,clustering_accuracy_mean,clustering_nmi_mean]
            results[dataset]['std'][:,index] = [classification_accuracy_std, clustering_accuracy_std, clustering_nmi_std]
            
            with open(result_file_path,'wb') as f:
                pickle.dump(results, f)

            print('p=%d'%p)
            print(50*'-')
            print('%.3f, %.3f'%(clustering_nmi_mean, clustering_accuracy_mean))
            print('%.3f'%classification_accuracy_mean)
        
if __name__ == '__main__':
    main()