'''
knn 
TODO 
- Tie break
'''

import numpy as np
import operator
from math import log10
from collections import Counter

def L1(v1,v2):
    return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])

def L2(v1,v2):
    return np.linalg.norm(v1 - v2)

def normalize(tr_data, test_data):
    for i in range(1,10):
        
        v = [j[i] for j in tr_data]
        tr_data_norm = v - np.mean(v)
        tr_data_norm = tr_data_norm / np.std(v)
        tr_data[:,i] = tr_data_norm
        
        v_t = [j[i] for j in test_data]
        test_data_norm = v_t - np.mean(v)
        test_data_norm = test_data_norm / np.std(v)
        test_data[:,i] = test_data_norm
      
    return tr_data, test_data

def find_nearest_knn(training_data, test_item, k, dist_function):
    id_label_distance = [] # values [id1, lab1, dist1], [id2, lab1, dist2]
    
    test_id = test_item[0]
    test_label = test_item[10]
    test_features = test_item[1:10]
        
    for item in training_data:
        id = item[0]
        label = item[10]
        features = item[1:10]
        
        if id == test_id:
#             if test_item is present in training_data we don't use it for classification
            continue;
        
        dist = dist_function(features, test_features)
        id_label_distance.append([id,label,dist])
    
    # sort id_label_distance by distance
    id_label_distance.sort(key=lambda x: x[2])
    
#     print id_label_distance
#     return top elements with most occurrence
#     print test_label, test_id
#     print [[i[0], i[1]] for i in id_label_distance[0:k]]
    
#     form counter of labels from id_label_distance
    nearst_counter = Counter([i[1] for i in id_label_distance[0:k]])
#     check if tie break needed    
    if len(nearst_counter.keys()) > 1 and nearst_counter.most_common(2)[0][1] == nearst_counter.most_common(2)[1][1]:
        # Equal number of records found. Get that number
        clash_counts = nearst_counter.most_common(1)[0][1]
        for i_l_d in id_label_distance[0:k]:
            # loop on sorted array and report forst number whose count is equal to cash count
            
            if nearst_counter[i_l_d[1]] == clash_counts:
                predict_label = i_l_d[1]
                break
                       
    else:
        # no tie breake needed
        predict_label = nearst_counter.most_common(1)[0][0] 
    
    
    
    return id_label_distance[0:k], predict_label==test_label

def run_knn(data, test_data, dist_function):
    k_set = {1:[],3:[],5:[],7:[]}
    
    for test_item in test_data:    
        k_set[1].append(find_nearest_knn(data, test_item, 1, dist_function)[1])
        k_set[3].append(find_nearest_knn(data, test_item, 3, dist_function)[1])
        k_set[5].append(find_nearest_knn(data, test_item, 5, dist_function)[1])
        k_set[7].append(find_nearest_knn(data, test_item, 7, dist_function)[1])
    
   
    print "k = 1 ",np.average(k_set[1])
    print "k = 3 ",np.average(k_set[3])
    print "k = 5 ",np.average(k_set[5])
    print "k = 7 ",np.average(k_set[7])

tr_data = np.genfromtxt("train.txt",delimiter=",",usecols=range(0 , 11))
test_data = np.genfromtxt("test.txt",delimiter=",",usecols=range(0 , 11))

normalize(tr_data, test_data)

print "\nKNN - L2 - Euclidean"
print "Test - "
run_knn(tr_data, test_data, L2)
print "Training - "
run_knn(tr_data, tr_data, L2)

print "\nKNN - L1 - Manhattan"
print "Test - "
run_knn(tr_data, test_data, L1)
print "Training - "
run_knn(tr_data, tr_data, L1)
