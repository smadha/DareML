'''
knn 
'''

import numpy as np
import scipy.stats
import operator
from math import log10
from collections import Counter

def find_nearest_knn(training_data, test_item, k):
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
        
        dist = np.linalg.norm(features - test_features)
        id_label_distance.append([id,label,dist])
    
    # sort id_label_distance by distance
    id_label_distance.sort(key=lambda x: x[2])
    
#     print id_label_distance
#     return top elements with most occurence
#     print test_label, test_id
#     print [[i[0], i[1]] for i in id_label_distance[0:k]]
    predict_label = Counter([i[1] for i in id_label_distance[0:k]]).most_common(1)[0][0]
    
    return id_label_distance[0:k], predict_label==test_label

def run_knn(data, test_data):
    k_set = {1:[],3:[],5:[],7:[]}
    
    for test_item in test_data:    
        k_set[1].append(find_nearest_knn(data, test_item, 1)[1])
        k_set[3].append(find_nearest_knn(data, test_item, 3)[1])
        k_set[5].append(find_nearest_knn(data, test_item, 5)[1])
        k_set[7].append(find_nearest_knn(data, test_item, 7)[1])
    
    print "1",np.average(k_set[1])
    print "3",np.average(k_set[3])
    print "5",np.average(k_set[5])
    print "7",np.average(k_set[7])

tr_data = np.genfromtxt("train.txt",delimiter=",",usecols=range(0 , 11))
test_data = np.genfromtxt("test.txt",delimiter=",",usecols=range(0 , 11))

print "Test - "
run_knn(tr_data, test_data)
print "Training - "
run_knn(tr_data, tr_data)



