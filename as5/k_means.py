'''
Implement k-means using random initialization for cluster centers. 
The algorithm should run until none of the cluster assignments are changed. 
Run the algorithm for different values of K -> {2, 3, 5}, and plot the clustering assignments by different colors 
and markers. (you need to report 6 plots, 3 for each dataset.)
'''
import numpy as np
from input_data_util import blob_data, circle_data, plot_scatter

def linear(x1):
    return x1

def poly(x1):
    return np.array([x1[0] ** 2, x1[1] ** 2, x1[0] ** 2 + x1[1] ** 2])


color_arr = ["red", "green", "blue", "yellow", "grey"]
def run_k_means(K, X, data_set_name, trans = linear):
    '''
    K -> num of clusters
    X -> all data points X[0] = [x_1, x_2]
    trans -> transformation kernel
    '''
    # Initialize mean randomly
    means = []
    for random_x in np.random.choice(len(X),K):
        means.append( trans(X[random_x]) )
    
    #store this value to determine when to stop
    avg_dist_with_prev = 10
    # dict of cluster to list of points in that cluster
    cluster = dict.fromkeys(range(K))
        
    while avg_dist_with_prev > 0.01:
        #store current mean to be used later
        prev_means = [] + means
        #initialize all clusters with 0 data points
        for k in cluster:
            cluster[k] = []
            
        # assign cluster to each point
        for x in X:
            dist_k = []
            for k in range(K):
                dist_k.append(np.linalg.norm(means[k] - trans(x) ))
            
            nearest_k = np.argmin(dist_k)
            cluster[nearest_k].append(x)
        
        dist_with_prev = []
        
        # find mean of all point in a cluster
        for k in range(K):
            trans_cluster_k = [trans(x_cluster_k) for x_cluster_k in cluster[k]]

            means[k] = np.mean(trans_cluster_k, axis = 0)
            
            # calculate distance with previous means
            dist_with_prev.append(np.linalg.norm(  means[k] -  prev_means[k] ))
        
        # average distance with prev
        avg_dist_with_prev = np.average(dist_with_prev)
        
    print data_set_name, "Cluster :", "\t".join([ "{0} - {1} points".format(cluster_id+1, len(points)) for cluster_id, points in cluster.iteritems()])
    plot_scatter( [(points, color_arr[cluster_id]) for cluster_id, points in cluster.iteritems()], "./kmeans/" + data_set_name + str(K), means )
        
    
if __name__ == '__main__':
    
    print "Running K-Means"
    for k in [2, 3, 5]:
        run_k_means(k, blob_data, "blob_data")
        run_k_means(k, circle_data, "circle_data") 
    
    print "Running K-Means with kernel"
    run_k_means(2, circle_data, "circle_data_kernel",poly)
    
