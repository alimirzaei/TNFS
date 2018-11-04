from scipy import io
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.manifold import TSNE
from RRFS import RRFS
from utils import evaluate_classification, evaluate_clustering, evaluate_reconstruction
import numpy as np
import pickle
import os
import methods as method_functions

def main():
    directory = 'results'    
    if not os.path.exists(directory):
        os.makedirs(directory)
    methods = ['udfs_score',  'laplacian_score', 'my_tsne', 'my_isomap']
    datasets = ['COIL20']#,'Yale','PCMAC','BASEHOCK','RELATHE','Prostate_GE' ,'Isolet', 'mnist']
    for method in methods:
        result_file_path = '%s/%s.pkl'%(directory, method)
        if(os.path.exists(result_file_path)):
            with open(result_file_path, 'rb') as f:
                results = pickle.load(f)    
        else:
            results={}
        
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

            idx = getattr(method_functions, method)(X, y) 
            # perform evaluation on clustering task
            
            percents = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 100]
            
            
            results[dataset]['mean'] = np.zeros((4, len(percents)))
            results[dataset]['std'] = np.zeros((4, len(percents)))
            results[dataset]['feature_ranking'] = idx
            for index,p in enumerate(percents):
                # obtain the dataset on the selected features
                num_fea = int(p*X.shape[1]/100)    # number of selected features
                selected_features = idx[:num_fea]
                selected_X = X[:, selected_features]

                (classification_accuracy_mean, classification_accuracy_std) = evaluate_classification(selected_X,y)
                (clustering_nmi_mean, clustering_nmi_std), (clustering_accuracy_mean, clustering_accuracy_std) = evaluate_clustering(selected_X, y)
                (reconstruction_mean, reconstruction_std) = evaluate_reconstruction(X, selected_features)
                
                results[dataset]['mean'][:,index] = [classification_accuracy_mean,clustering_accuracy_mean,clustering_nmi_mean, reconstruction_mean]
                results[dataset]['std'][:,index] = [classification_accuracy_std, clustering_accuracy_std, clustering_nmi_std, reconstruction_std]
                
                with open(result_file_path,'wb') as f:
                    pickle.dump(results, f)

                print('p=%d'%p)
                print(50*'-')
                print('%.3f, %.3f'%(clustering_nmi_mean, clustering_accuracy_mean))
                print('%.3f'%classification_accuracy_mean)
        
if __name__ == '__main__':
    main()