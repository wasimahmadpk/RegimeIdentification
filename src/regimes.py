import random
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
import matplotlib.ticker as ticker
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
     
    if dtype == 'real':
          
        # Plot regimes in real data
        colors = ['r', 'g', 'b', 'y', 'c']

        t = np.arange(0, cluster_idx[-1]+winsize)
        start = 0

        plt.figure(figsize=(15, 4))
        col = ['teal', 'slategrey', 'goldenrod']
        mark = ['-', '--', '.-.']

        ax = data[plot_var].plot(cmap='viridis_r', figsize=(9, 3), linewidth=0.66)
        plt.legend(plot_var)

        prev = clusters[0]
        
        # Create a dictionary to store regime names and starting points
        regime_starting_points = {}
        for c in range(len(cluster_idx)):

            curr = clusters[c]
            val = cluster_idx[c]
            rcp = 0
            # print(f'Index data: {data.index[val]}, Index value: {val}')
            regime_starting_points[f'Regime {c+1}'] = data.index[val]
            if prev != curr:
                plt.axvline(x=val, color='black', linestyle='--', linewidth=0.75)
                # max_vals, max_indices = data.loc[rcp: val].max(), data.loc[rcp: val].idxmax()
                # plot_marker(data, max_vals, max_indices)
                rcp = val
                prev = curr
            
            if clusters[c] == 0:
                plt.axvspan(val, val+winsize, color='gray', alpha=0.15)
            
            if clusters[c] == 1:
                plt.axvspan(val, val+winsize, color='white', alpha=0.15)    #random.choice(['green', 'blue', 'red'])
            
            if clusters[c] == 2:
                plt.axvspan(val, val+winsize, color='green', alpha=0.15)    #data.index[val], data.index[val+winsize]
            
            if clusters[c] == 3:
                plt.axvspan(val, val+winsize, color='blue', alpha=0.15)  
        print(regime_starting_points)
        
        if dtype!='real':
            max_vals, max_indices = data.loc[val: ].max(), data.loc[val: ].idxmax()
            plot_marker(data, max_vals, max_indices)
            plt.axvline(x=365, color='red')
            plt.text(305, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
            plt.axvline(x=730, color='red')
            plt.text(670, 1.10, 'Change Point', fontsize=9.0, fontweight='bold')
            plt.axvline(x=1095, color='red')
        
        plt.ylim(0, 1.5)
        # plt.gcf().autofmt_xdate()
        plt.legend(plot_var, loc='upper right', frameon=True, ncol=3)
        plt.xlabel('Data points')
        plt.ylabel('Values')
        ax.tick_params(length=4)
        plt.grid(False)
        # plt.savefig("../res/climate_regimes.pdf", bbox_inches='tight')
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