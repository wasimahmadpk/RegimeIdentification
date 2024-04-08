import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def regularize_covmat(cov_matrix, reg_param):
    """Regularize a covariance matrix by adding a scaled identity matrix."""
    n_features = cov_matrix.shape[0]
    reg_matrix = reg_param * np.eye(n_features)
    regularized_cov_matrix = cov_matrix + reg_matrix
    return regularized_cov_matrix

# # Example usage
# cov_matrix = np.array([[2, 1], [1, 3]])  # Example covariance matrix
# reg_param = 0.1  # Regularization parameter
# regularized_cov_matrix = regularize_covariance_matrix(cov_matrix, reg_param)
# print("Original Covariance Matrix:")
# print(cov_matrix)
# print("Regularized Covariance Matrix:")
# print(regularized_cov_matrix)

def getSPDMs(data, wsize, reg):
    
    winsize = wsize
    start = 0
    flat_cov_mat = []
    cov_mat = []
    columns = data.columns
    dim = len(columns) - 1
    cluster_idx = []
 
    while start+wsize < len(data)-1:
        cluster_idx.append(start)

        data_batch = data[start: start + winsize]
        ls_data_batch = []
        
        for i in range(len(columns)):
            ls_data_batch.append(data_batch[columns[i]].values.tolist())

        cov = regularize_covmat(np.cov(np.array(ls_data_batch)), reg_param=reg)
        upper = np.triu(cov, k=0)
        mask = np.triu_indices(dim)
        newupp = list(upper[mask])
        upp = list(upper[upper!=0])
        
#         mean_v = list(np.mean(np.array(ls_data_batch), axis=1))
        
        feat = stats.describe(np.array(ls_data_batch), axis=1)
        mean_val = feat.mean.tolist()
        skewness = feat.skewness.tolist()
        kurtosis = feat.kurtosis.tolist()
        
#         plt.plot(helper.normalize(newupp, 'std'))
#         plt.show()
        mix_feat = newupp 
        flat_cov_mat.append(mix_feat)
        cov_mat.append(cov)
        start = start + winsize

    return flat_cov_mat, cov_mat, cluster_idx