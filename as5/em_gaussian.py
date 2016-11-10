
import numpy as np
from input_data_util import blob_data, plot_scatter
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

color_arr = ["red", "green", "blue", "yellow", "grey"]

def plot_line(x_y, name):
    '''
    plot list of line in a figure with diff colors 
    '''
    for idx,(x,y) in enumerate(x_y):
        plt.plot(x,y, markersize=5, color=color_arr[idx])
        
    plt.figtext(.02, .02, name)
    plt.savefig(name)
    plt.close()

def calc_mean(X, p_ik, k):
    '''
    X -> All data points shape -> (N ,2)
    p_ik -> Probability of X_i in cluster k. shape -> (N ,K)
    
    return -> mean of shape (1,2)
    '''    
    return np.sum(np.transpose([p_ik[:, k]]) * X , axis=0) / np.sum(p_ik[:, k])

def calc_var(X, p_ik, mean_k, k):
    '''
    X -> All data points shape -> (N ,2)
    p_ik -> Probability of X_i in cluster k. shape -> (N ,K)
    mean_k -> mean of k cluster
    return -> cocovar matrix of shape (2,2) for cluster k
    '''
    N_k = np.sum(p_ik[:, k])
    X_minus_mean = X - mean_k
    
    sum_var = np.zeros((2, 2))
    for i in range(len(X_minus_mean)) :        
        sum_var += p_ik[i][k] * np.dot(np.transpose([X_minus_mean[i]]), np.array([X_minus_mean[i]]))
    
    return sum_var / N_k

def run_em_gaussian(K, X, data_set_name):
    '''
    K -> num of clusters
    X -> all data points X[0] = [x_1, x_2]
    '''
    log_like = []
    # Initialize mean randomly
    means = []
    for random_x in np.random.choice(len(X), K):
        means.append(X[random_x])
    
    # Initialize covar with ones
    covar = []
    for k in range(K):
        covar.append(np.cov(X, rowvar=False) )
    
    # store this value to determine when to stop
    avg_dist_with_prev = 10
            
    # Probability of X_i in cluster k. shape -> (N ,K)
    p_ik = []
    prior = np.array( [ 1.0/K for k in range(K)] ) 
    while avg_dist_with_prev > 0.001:
        # store current mean to be used later
        prev_means = [] + means
        
        p_ik = []
        n_ik = []
        # calculate p_ik for each point
        for idx, x in enumerate(X):
            n_ik.append([])
            for k in range(K):
                n_ik[idx].append(multivariate_normal.pdf(x, mean=means[k], cov=covar[k]))
            
            sum_n_ik = np.sum(n_ik[idx] * prior)
            p_ik.append([ n / sum_n_ik for n in n_ik[idx] * prior ])
        
        dist_with_prev = []
        
        p_ik = np.array(p_ik)
        n_ik = np.array(n_ik)
        # find mean and covar of all clusters
        for k in range(K):
            means[k] = calc_mean(X, p_ik, k)
            covar[k] = calc_var(X, p_ik, means[k], k)
            
            # calculate distance with previous means
            dist_with_prev.append(np.linalg.norm(means[k] - prev_means[k]))
        
        # average distance with prev
        avg_dist_with_prev = np.average(dist_with_prev)
        
        prior = np.array( [ sum(p_ik[:,k])/len(X) for k in range(K)] )
        
        # calculate log likelihood of whole data
        log_like.append(sum( np.log([sum(x * prior) for x in n_ik]) ))
        
    
    
    
    # dict of cluster to list of points in that cluster
    cluster = dict.fromkeys(range(K))
    # initialize all clusters with 0 data points
    for k in cluster:
        cluster[k] = []
    # assign k to X if p_ik is max for X  
    for i in range(len(X)):
        k = np.argmax(p_ik[i])
        cluster[k].append(X[i])
        
    print means, covar
    print data_set_name, "Cluster :", "\t".join([ "{0} - {1} points".format(cluster_id + 1, len(points)) for cluster_id, points in cluster.iteritems()])
    plot_scatter([(points, color_arr[cluster_id]) for cluster_id, points in cluster.iteritems()], "./kmeans/" + data_set_name , means)
    
    return (range(len(log_like)), [l+100 for l in log_like])

def main_fn():
    print "Running EM"
    x_y = []
    for i in range(5):
        x_y.append(run_em_gaussian(3, blob_data, "blob_data_EM_{0}".format(i))) 
        
    plot_line( x_y , "./kmeans/log_like_blob_data_EM")
    
if __name__ == '__main__':
    main_fn()
    
    
