import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from spdms import getSPDMs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
plt.rcParams['figure.dpi'] = 200

from sklearn.cluster import KMeans
from pyriemann.clustering import Kmeans
from sklearn import metrics
from scipy.spatial.distance import cdist



def get_regimes(data, wsize, dist_metric):
    
        covmat, covar, cluster_idx = getSPDMs(data, wsize)
        if dist_metric == 'Euclidean':
        
            kmeans = KMeans(n_clusters=3, random_state=0, n_init=1).fit(covmat)
            clusters = list(kmeans.labels_)
            print(f"Clusters: {list(kmeans.labels_)}")
    
#       for k in K:
#             kmeans = KMeans(n_clusters=k, random_state=0, n_init=1).fit(covmat)
#             clusters = list(kmeans.labels_)
#             print(f"Clusters: {list(kmeans.labels_)}")
        
#             distortions.append(sum(np.min(cdist(covmat, kmeans.cluster_centers_, 'mahalanobis'), axis=1)) / np.array(covmat).shape[0])
#             inertias.append(kmeans.inertia_)
#             mapping1[k] = sum(np.min(cdist(covmat, kmeans.cluster_centers_, 'mahalanobis'), axis=1)) / np.array(covmat).shape[0]
#             mapping2[k] = kmeans.inertia_
        
#         #   The elbow method for optimal number of clusters
#         plt.plot(K, inertias, 'bx-')
#         plt.xlabel('Values of K')
#         plt.ylabel('Distortion')
#         plt.title('The Elbow Method using Distortion')
#         plt.show()
    
        else:
#           clusters = cluster(np.array(covmat))
            clusters = pyriemann_clusters(np.array(covar))
#     
            print(f"Clusters indecis: {cluster_idx}")
        return clusters, cluster_idx
    

    # Time series segmentation with K-Means (Riemanian metric as a distance measure)

def pyriemann_clusters(data):
    
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        
        kmeans = Kmeans(3, metric='riemann', tol=1e-3, init='random')
        kmeans.fit(data)
        labels = kmeans.predict(data)
        centroids = kmeans.centroids
        print(labels)


    # Find optimal number of regimes
    
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


def plot_regimes(data, clusters, cluster_idx, winsize, dtype='real'):
     
    if dtype == 'real':
          
        # # Plot regimes in real data

        # toplot = [ 'rain','strain_ns_corrected', 'tides_ns', 'temperature_outside', 'pressure_outside', 'gw_west']
        toplot = ['gw_mb', 'gw_sg', 'temperature_outside', 'strain_ew_corrected', 'strain_ns_corrected']
        # toplot = ['gw_mb', 'gw_sg', 'gw_west', 'gw_south', 'strain_ns_corrected']
        # toplot = ['Hs', 'P', 'W' ]
        colors = ['r', 'g', 'b', 'y', 'c']

        t = np.arange(0, cluster_idx[-1]+winsize)
        start = 0


        plt.figure(figsize=(12, 4))
        col = ['teal', 'slategrey', 'goldenrod']
        mark = ['-', '--', '.-.']

        # for i, v in enumerate(toplot):
        #         data.plot(use_index=True, figsize=(10, 3), linewidth=0.75)
        #         plt.plot(data[v], mark[i], color=col[i])
        #         plt.plot(t[start: start+winsize], data[toplot[i]].values[start: start + winsize], colors[i]+marker)
        #         plt.plot(t[start: start + winsize], data[toplot[i+1]].values[start: start + winsize], color)
        #         plt.plot(t[start: start + winsize], data[toplot[i+2]].values[start: start + winsize], color)

        data[toplot].plot(use_index=True, cmap='tab10', figsize=(9, 3), linewidth=0.75)
        plt.legend(toplot)
        for c in range(len(cluster_idx)):

            val = cluster_idx[c]
            if clusters[c] == 0:
                for v in range(winsize):
                    plt.axvline(val+v, color="gray", alpha=0.1)
            if clusters[c] == 1:
                for v in range(winsize):
                    plt.axvline(val+v, color="white", alpha=0.00)
            if clusters[c] == 2:
                for v in range(winsize):
                    plt.axvline(val+v, color="gray", alpha=0.15)
            if c not in [0, 3]:
                plt.axvline(x=val, color='black', linestyle='--', linewidth=0.75)
        # plt.axvline(x=365, color='red')
        # plt.text(305, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        # plt.axvline(x=730, color='red')
        # plt.text(670, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.ylim(0, 1.35)
        # plt.gcf().autofmt_xdate()
        # plt.legend(['$Z_{1}$', '$Z_{2}$', '$Z_{3}$'], loc='upper left', fontsize=6, prop=dict(weight='bold'))
        plt.legend(['GW$_{mb}$', 'GW$_{sg}$', 'T', 'Strain$_{ew}$', 'Strain$_{ns}$'], loc='upper right', frameon=True, ncol=5)
        plt.xlabel('')
        plt.ylabel('normalized values')
        plt.savefig("../res/georegimes2.pdf", bbox_inches='tight')
        plt.show()

    else:
        # Plot regimes in synthetic data

        toplot = ['Z1', 'Z3', 'Z5']
        # toplot = ['Hs', 'P', 'W' ]
        colors = ['r', 'g', 'b', 'y', 'c']

        t = np.arange(0, cluster_idx[-1]+winsize)
        start = 0

        # for c in range(len(clusters)):
            
        #     if clusters[c] == 0:
        #             marker = '-'
        #     elif clusters[c] == 1:
        #             marker = '-'
        #     elif clusters[c] == 2:
        #             marker = '-'
        #     for i in toplot:
                
        #         data[i].plot(use_index=True)
        #         plt.legend(toplot)
        # #         plt.plot(t[start: start+winsize], data[toplot[i]].values[start: start + winsize], colors[i]+marker)
        # #         plt.plot(t[start: start + winsize], data[toplot[i+1]].values[start: start + winsize], color)
        # #         plt.plot(t[start: start + winsize], data[toplot[i+2]].values[start: start + winsize], color)
                
        #     start = start + winsize

        plt.figure(figsize=(6, 3))
        col = ['teal', 'slategrey', 'goldenrod']
        mark = ['-', '--', '.-.']
        for i, v in enumerate(toplot):
                # data.plot(use_index=True, figsize=(10, 3), linewidth=0.75, alpha=0.66, color=['green', 'blue', 'red'])
                plt.plot(data[v], mark[i], color=col[i])
        #         plt.plot(t[start: start+winsize], data[toplot[i]].values[start: start + winsize], colors[i]+marker)
        #         plt.plot(t[start: start + winsize], data[toplot[i+1]].values[start: start + winsize], color)
        #         plt.plot(t[start: start + winsize], data[toplot[i+2]].values[start: start + winsize], color)


        plt.legend(toplot)
        for c in range(len(cluster_idx)):
                
            val = cluster_idx[c]
            if clusters[c] == 0:
                for v in range(winsize):
                    plt.axvline(val+v, color="green", alpha=0.025)
            if clusters[c] == 1:
                for v in range(winsize):
                    plt.axvline(val+v, color="white", alpha=0.00)
            if clusters[c] == 2:
                for v in range(winsize):
                    plt.axvline(val+v, color="red", alpha=0.025)
        plt.axvline(x=364, color='red')
        # plt.text(305, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.axvline(x=720, color='red')
        # plt.text(670, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
        plt.ylim(0, 1.3)
        # plt.gcf().autofmt_xdate()
        plt.legend(['$Z_{1}$', '$Z_{2}$', '$Z_{3}$'], loc='upper left', fontsize=50, prop=dict(weight='bold'))
        # plt.title("Euclidean", fontsize=15)
        plt.ylabel("window=90", fontsize=15)
        # plt.xlabel('data points', fontsize=10)
        plt.savefig("../res/synwin90EE.pdf")

        plt.show()


# Normalization (Standard)
def normalize(data, type='minmax'):

    if type=='std':
        return (np.array(data) - np.mean(data))/np.std(data)
        
    elif type=='minmax':
        return (np.array(data) - np.min(data))/(np.max(data) - np.min(data))

class main:

    # data = pd.read_csv(r'../datasets/moxa_data_D.csv')
    data = pd.read_csv(r'../datasets/synthetic_datac.csv')


    # start, end = '2016-06-15', '2016-12-30'
    # mask = (data['DateTime'] > '2014-11-01') & (data['DateTime'] <= '2015-12-28')
    # data = data.loc[mask]
    # data = data.fillna(method='pad')
    # data = data.set_index('DateTime')
    # data = data[start: end]
    data.info()
    data = data.apply(normalize)
    # data.head(100)

    winsize = 90 # 155
    metricE = 'Euclidean'
    metricR = 'Riemannian'
    clusters, cluster_idx = get_regimes(data, winsize, metricE)

    clusters_extended = []
    for i in range(len(clusters)):

        val = clusters[i]
        for j in range(winsize):
            clusters_extended.append(val)
        
    print(len(clusters_extended), data.shape)

    datanew = data.iloc[:len(clusters_extended), :].copy()
    datanew['Clusters'] = clusters_extended

    toplot = ['gw_mb', 'gw_sg', 'temperature_outside', 'strain_ew_corrected', 'strain_ns_corrected']
    plot_regimes(data, clusters, cluster_idx, winsize, dtype='synthetic')

    
    
    