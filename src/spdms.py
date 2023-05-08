import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def getSPDMs(data, wsize):
    
    winsize = wsize
    start = 0
    covmat = []
    covar = []
    columns = data.columns
    dim = len(columns) - 1
    cluster_idx = []
 
    while start+winsize < len(data)-1:
        cluster_idx.append(start)
#         print(f"Data shape: {data.shape}")
        data_batch = data[start: start + winsize]
#         print(f"Data batch: {data_batch.shape}")
        ls_data_batch = []
        
        for i in range(len(columns)):
            ls_data_batch.append(data_batch[columns[i]].values.tolist())

        cov = np.cov(np.array(ls_data_batch))
#         print(f"Covariance of {columns[14]} with other variables: {cov[14]}")
#         flat_cov = np.concatenate(cov).ravel().tolist()
        upper = np.triu(cov, k=0)
#         print(f"Length of Cov matrix: {len(upper[upper!=0])}")
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
#         print(f"Length of features pool: {len(mix_feat)}")
        covmat.append(mix_feat)
        covar.append(cov)
#         print(covmat)
        start = start + winsize

    return covmat, covar, cluster_idx