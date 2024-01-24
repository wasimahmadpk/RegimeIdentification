import parameters
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from spdms import getSPDMs
from sklearn import metrics
# from sklearn import metrics
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from dimreduce import reduce_dimension
from pyriemann.clustering import Kmeans
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller

plt.rcParams['figure.dpi'] = 200


# Parameters
pars = parameters.get_syn_params()
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
plot_path = pars.get("plot_path")

def pyriemann_clusters(data, k):
    
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    kmeans = Kmeans(k, metric='riemann', tol=1e-3, init='random')
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.centroids
    
#     for k in K:
#         kmeans = KMeans(k, 'riemann', tol=1e-3, init='random')
#         kmeans.fit(data)
#         labels = kmeans.predict(data)
#         centroids = kmeans.centroids
#         print(labels)
        
#         distortions.append(sum(np.min(cdist(data, kmeans.centroids, 'euclidean'), axis=1)) / np.array(data).shape[0])
#         inertias.append(kmeans.inertia_)
#         mapping1[k] = sum(np.min(cdist(data, kmeans.centroids, 'euclidean'), axis=1)) / np.array(data).shape[0]
#         mapping2[k] = kmeans.inertia_
        
#     #   The elbow method for optimal number of clusters
#     plt.plot(K, inertias, 'bx-')
#     plt.xlabel('Values of K')
#     plt.ylabel('Distortion')
#     plt.title('The Elbow Method using Distortion')
#     plt.show()
    
    return labels


def get_regimes(data, wsize, k, dist_metric, dim='full'):

    flat_cov_mat, cov_mat, cluster_idx = getSPDMs(data, wsize)    
    
    if dim != 'full':
        assert int(dim) < len(data.columns), f"Reduced dimension:{int(dim)} is greater than full dimension:{len(data.columns)} ."
        cov_mat = np.transpose([cov_mat])
        rdata = reduce_dimension(cov_mat, int(dim)) 
        cov_mat = np.transpose(rdata[:, :, :, 0])
        
    if dist_metric == 'Euclidean':
            
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=1).fit(flat_cov_mat)
        clusters = list(kmeans.labels_)
        
    else:
        clusters = pyriemann_clusters(np.array(cov_mat), k)     

    clusters_extended = []
    for i in range(len(clusters)):

        val = clusters[i]
        for j in range(slidingwin_size):
            clusters_extended.append(val)
        
        # newdf = data.iloc[:len(clusters_extended), :].copy()
        # newdf['Clusters'] = clusters_extended

    dfs = []
        # for c in range(len(list(set(clusters)))):
        #     dfs.append(newdf.loc[newdf['Clusters'] == list(set(clusters))[c]])
    zip_regimes = list(zip(clusters, cluster_idx))
    print("Regimes:" + " ".join(map(str, zip_regimes)))

    return clusters, cluster_idx, dfs
   

def get_reduced_set(df):
    
    corr = data.corr()
    cls = corr.iloc[0][:].values.tolist()
    selected_idx = np.where(cls>0.50)[0].tolist()

    reduced_df = df.iloc[:, selected_idx].copy()
    return reduced_df


def plot_regimes(data, plot_var, clusters, cluster_idx, winsize, dtype='real'):
     
    if dtype == 'real':
          
        # Plot regimes in real data
        colors = ['r', 'g', 'b', 'y', 'c']

        t = np.arange(0, cluster_idx[-1]+winsize)
        start = 0

        plt.figure(figsize=(12, 4))
        col = ['teal', 'slategrey', 'goldenrod']
        mark = ['-', '--', '.-.']

        data[plot_var].plot(use_index=True, cmap='tab10', figsize=(9, 3), linewidth=0.66)
        plt.legend(plot_var)

        prev = clusters[0]
        for c in range(len(cluster_idx)):

            curr = clusters[c]
            val = cluster_idx[c]
            
            if prev != curr:
                plt.axvline(x=val, color='red', linestyle='--', linewidth=0.75)
                prev = curr
            
            if clusters[c] == 0:
                plt.axvspan(val, val+winsize, color='white', alpha=0.5)
            
            if clusters[c] == 1:
                plt.axvspan(val, val+winsize, color='gray', alpha=0.6)    
            
            if clusters[c] == 2:
                plt.axvspan(val, val+winsize, color='blue', alpha=0.2)    
            
            if clusters[c] == 3:
                plt.axvspan(val, val+winsize, color='green', alpha=0.5)  
              
        # plt.axvline(x=365, color='red')
        # plt.text(305, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        # plt.axvline(x=730, color='red')
        # plt.text(670, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.ylim(0, 1.35)
        # plt.gcf().autofmt_xdate()
        # plt.legend(['GW$_{mb}$', 'GW$_{sg}$', 'T', 'Strain$_{ew}$', 'Strain$_{ns}$'], loc='upper right', frameon=True, ncol=5)
        plt.legend(plot_var, loc='upper right', frameon=True, ncol=5)
        plt.xlabel('')
        plt.ylabel('normalized values')
        # plt.savefig("../res/georegimes2.pdf", bbox_inches='tight')
        # Convert month number to month name
        plt.show()

    else:
        # Plot regimes in synthetic data

        plot_var = ['Z1', 'Z3', 'Z5']
        colors = ['r', 'g', 'b', 'y', 'c']

        t = np.arange(0, cluster_idx[-1]+winsize)
        start = 0

        plt.figure(figsize=(6, 3))
        col = ['teal', 'slategrey', 'goldenrod']
        mark = ['-', '--', '.-.']

        for i, v in enumerate(plot_var):
            plt.plot(data[v], mark[i], color=col[i])
   
        plt.legend(plot_var)
        for c in range(len(cluster_idx)):
                
            val = cluster_idx[c]
            # print(f'{val} to {val+winsize}')
            if clusters[c] == 0:
                plt.axvspan(val, val+winsize, facecolor='green', alpha=0.15)    
            if clusters[c] == 1:
                plt.axvspan(val, val+winsize, facecolor='white', alpha=0.25)  
            if clusters[c] == 2:
                plt.axvspan(val, val+winsize, facecolor='red', alpha=0.15)  
            if clusters[c] == 3:
                plt.axvspan(val, val+winsize, facecolor='yellow', alpha=0.15)  
        
        plt.axvline(x=364, color='red')
        # plt.text(305, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.axvline(x=720, color='red')
        # plt.text(670, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.ylim(0, 1.3)
        plt.legend(['$Z_{1}$', '$Z_{2}$', '$Z_{3}$'], loc='upper left', fontsize=50, prop=dict(weight='bold'))
        # plt.title("Euclidean", fontsize=15)
        plt.ylabel("window=90", fontsize=15)
        # plt.xlabel('data points', fontsize=10)
        # plt.savefig("../res/synwin90EE.pdf")

        plt.show()