import parameters
import numpy as np
import pandas as pd
from spdms import getSPDMs
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dimreduce import reduce_dimension
from pyriemann.clustering import Kmeans
from pyriemann.utils.covariance import covariances_X
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from yellowbrick.cluster import KElbowVisualizer

import warnings
from contextlib import suppress

plt.rcParams['figure.dpi'] = 200

# Parameters
pars = parameters.get_syn_params()
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
plot_path = pars.get("plot_path")


def read_file(file_path):
    # Check the file extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the file based on the extension
    if file_extension in ['xls', 'xlsx']:
        data = pd.read_excel(file_path)
    elif file_extension == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension == 'txt':
        data = pd.read_csv(file_path, sep=' ')
    elif file_extension == 'h5':
        data = pd.read_hdf(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return data

def find_time_related_columns(df):

    # List of potential time-related column names
    time_column_names = ['date', 'Date', 'timestep', 'timestamp']

    # Check for the presence of time-related columns
    time_columns = [col for col in df.columns if col.lower() in time_column_names]

    # Return a tuple with a boolean and the list of time-related columns
    return bool(time_columns), time_columns


def shift_and_fill_mean(df, n):
    """
    Shifts the values in a single-column DataFrame and fills missing values with the mean.

    Parameters:
    - df: Original DataFrame with a single column.
    - n: Number of points to shift.

    Returns:
    - new_df: DataFrame with two columns, where missing values are filled with the mean.
    """

    # Create a new DataFrame with two columns by shifting the values
    new_df = pd.DataFrame({
        'col1': df[df.columns[0]].shift(-n),
        'col2': df[df.columns[0]]
    })

    # Fill NaN values in both columns with the mean of the original column
    new_df.fillna(df[df.columns[0]].mean(), inplace=True)

    return new_df


def find_optimal_k(data):
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Instantiate the clustering model and visualizer
    model = KMeans()

    # Suppress exceptions
    with suppress(AttributeError):

        plt.figure(figsize=(5,3))
        visualizer = KElbowVisualizer(model, k=(2, 6), metric='calinski_harabasz', timings=False)
        visualizer.fit(data)  # Fit the data to the visualizer
        plt.close()

        # Retrieve the optimal k
        optimal_k = visualizer.elbow_value_
        print(f'Optimal regimes: {optimal_k}')
        return optimal_k

def pyriemann_clusters(data, k):
    
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    kmeans = Kmeans(k, metric='riemann', tol=1e-3, init='random')
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.centroids
    
    return labels


def get_regimes(data, wsize, dist_metric, k=None, dim='full'):

    reg_param = 0.1
    flat_cov_mat, cov_mat, cluster_idx = getSPDMs(data, wsize, reg_param)    
    
    if dim.lower() != 'full':
        assert int(dim) < len(data.columns), f"Reduced dimension:{int(dim)} is greater than full dimension:{len(data.columns)} ."
        cov_mat = np.transpose([cov_mat])
        rdata = reduce_dimension(cov_mat, int(dim)) 
        cov_mat = np.transpose(rdata[:, :, :, 0])
    
    if k is None:
        k = find_optimal_k(data)
        
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
    prev = ''
    adjusted_idx = []
    for i in range(len(clusters)):
        if prev == '':
            adjusted_idx.append(cluster_idx[i])
            prev = clusters[0]
        else:
            if prev != clusters[i]:
                adjusted_idx.append(cluster_idx[i])
                prev = clusters[i]
    
    zip_regimes = list(zip(clusters, cluster_idx))
    print("Regimes:" + " ".join(map(str, zip_regimes)))

    return clusters, cluster_idx, adjusted_idx, dfs
   

def get_reduced_set(df):
    
    corr = data.corr()
    cls = corr.iloc[0][:].values.tolist()
    selected_idx = np.where(cls>0.50)[0].tolist()

    reduced_df = df.iloc[:, selected_idx].copy()
    return reduced_df

def plot_marker(df, max_val_lst, dict):

    counter = 0
    for column, max_index in dict.items():
        plt.scatter(max_index, df.loc[max_index, column], color='red') #label=f'Max {column}'
        counter = counter + 1



def visualize(data, plot_var, clusters, cluster_idx, winsize, dtype='real'):
    """
    Visualize detected regimes in time series data.
    
    Parameters:
        data (pd.DataFrame): The dataset containing time series.
        plot_var (list): List of variables to plot.
        clusters (list): Assigned cluster labels for each window.
        cluster_idx (list): Start indices for each window.
        winsize (int): Window size used in regime detection.
        dtype (str): 'real' for real-world data, 'synthetic' for synthetic data.
    """
    
    regime_colors = ['teal', 'white','slategrey', 'goldenrod'] #['green', 'white', 'gray', 'blue', 'yellow']
    
    if dtype == 'real':
        plt.figure(figsize=(15, 4))
        ax = data[plot_var].plot(figsize=(9, 3), linewidth=0.66)
        plt.legend(plot_var)

        prev_cluster = clusters[0]
        regime_start_points = {}

        for idx, val in enumerate(cluster_idx):
            current_cluster = clusters[idx]
            regime_start_points[f'Regime {idx+1}'] = data.index[val]

            if prev_cluster != current_cluster:
                plt.axvline(x=val, color='red', linestyle='--', linewidth=0.75)
                prev_cluster = current_cluster
            
            plt.axvspan(val, val + winsize, color=regime_colors[current_cluster], alpha=0.15)

        print("Regime Starting Points:")
        print(regime_start_points)


        plt.ylim(0, 1.25)
        plt.xlabel('Data points')
        plt.ylabel('Values')
        plt.legend(plot_var, loc='upper left', ncol=1, frameon=True)
        plt.grid(False)
        plt.savefig("../res/hurricane_anomaly.pdf", bbox_inches='tight', dpi=600, format='pdf')

        plt.show()

    else:  # synthetic data
        plt.figure(figsize=(6, 3))
        markers = ['-', '--', '-.']
        colors = ['teal', 'slategrey', 'goldenrod']
        synthetic_vars = ['Z1', 'Z3', 'Z5']

        for i, v in enumerate(synthetic_vars):
            plt.plot(data[v], markers[i], color=colors[i])

        plt.legend(synthetic_vars)

        for idx, val in enumerate(cluster_idx):
            plt.axvspan(val, val + winsize, facecolor=regime_colors[clusters[idx]], alpha=0.15)

        # Draw ground truth change points
        plt.axvline(x=364, color='red')
        plt.axvline(x=720, color='red')

        plt.legend(['$Z_{1}$', '$Z_{3}$', '$Z_{5}$'], loc='upper left', fontsize=10, prop=dict(weight='bold'))
        plt.ylabel(f"Window={winsize}", fontsize=15)
        plt.show()
