import parameters
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from spdms import getSPDMs
# from sklearn import metrics
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
plt.rcParams['figure.dpi'] = 200


from sklearn import metrics
from scipy.spatial.distance import cdist


# import autograd.numpy as np
from functools import reduce
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers.trust_regions import TrustRegions
import pymanopt



# Parameters
pars = parameters.get_syn_params()
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
plot_path = pars.get("plot_path")


def create_cost_and_derivatives(manifold, A):
    @pymanopt.function.autograd(manifold)
    def cost(x):
        T = A.shape[2]
        c = 0
        
        for i in range(T):
            c = c - (reduce(np.dot, [x.T, A[:,:,i], x, x.T, A[:,:,i], x])).trace()
        
        return c
    
    @pymanopt.function.autograd(manifold)
    def egrad(x):
        T = A.shape[2]
        gr = np.zeros(x.shape)
        
        for i in range(T):
            gr = gr - 4 * reduce(np.dot, [A[:,:,i], x, x.T, A[:,:,i], x])
        
        return gr
    return cost, egrad

def ManoptOptimization(A,m):
    n = A.shape[0]
    T = A.shape[2]
    manifold = Stiefel(n,m,k=1)

    cost,egrad = create_cost_and_derivatives(manifold, A)
    
    problem = Problem(manifold=manifold, cost = cost, euclidean_gradient= egrad)
    
    solver = TrustRegions()
    print('# Start optimization using solver: trustregion')
    Xopt = solver.run(problem)
    
    return Xopt

def reduce_dimension(covseqs, m):
    
    n,n,T,L = covseqs.shape    
    traj = np.zeros((m, m, T, L))
    B = np.zeros((n,m))
    
    for j in range(L):
        B = ManoptOptimization(covseqs[:,:,:,j], m)
        for i in range(T):
            traj[:,:,i,j] = reduce(np.dot, [B.point.T, covseqs[:,:,i,j], B.point])
            
    return traj 
                