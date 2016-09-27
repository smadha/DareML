'''
http://www.simafore.com/blog/bid/107702/2-ways-of-using-Naive-Bayes-classification-for-numeric-attributes
Attribute Information:
1. Id number: 1 to 214 
2. RI: refractive index 
3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10) 
4. Mg: Magnesium 
5. Al: Aluminum 
6. Si: Silicon 
7. K: Potassium 
8. Ca: Calcium 
9. Ba: Barium 
10. Fe: Iron 
11. Type of glass: (class attribute) 
-- 1 building_windows_float_processed 
-- 2 building_windows_non_float_processed 
-- 3 vehicle_windows_float_processed 
-- 4 vehicle_windows_non_float_processed (none in this database) 
-- 5 containers 
-- 6 tableware 
-- 7 headlamps

Probability while training with smoothing-
P(feature|class) - calculate mean and sigma for all feature and class combination
                   calculate probability as per normal distribution     

Probability while classifying with smoothing-
P(class|features) = [ P(feature1|class) * P(feature2|class) * .. * P(feature n|class) ]* P(class)
'''

import numpy as np
import scipy.stats
import operator
from math import log10

def classify(data_file, label_to_mean, label_to_std, label_to_pdf_to_feature, labels, label_to_prior):
    test_data = np.genfromtxt(data_file,delimiter=",",usecols=range(1 , 10))
    test_labels = np.genfromtxt(data_file,delimiter=",",usecols=range(10 , 11))
    
    accuracy = []
    for correct_label,features in zip(test_labels, test_data):
        label_prob ={}
        for label in labels:
            label_prob[label] = 0
            for i,feature in enumerate(features):
                mean = label_to_mean[label][i]
                std = label_to_std[label][i]
                if mean == 0 and std ==0: # no evidence in training data
                    if not feature == 0: # if found in test data penalize this class
                        label_prob[label] = -999
                    # else ignore this feature while classifying
                else:
                    label_prob[label] = label_prob[label] + log10(label_to_pdf_to_feature[label][i].pdf(feature))
            ## add prior probability of label
            label_prob[label] = label_prob[label]  + log10(label_to_prior[label])
            
        if correct_label == max_relation(label_prob.iteritems(), key=operator.itemgetter(1))[0]:
            accuracy.append(1)
        else:
            accuracy.append(0)
#         print label_prob
#         print correct_label, max_relation(label_prob.iteritems(), key=operator.itemgetter(1))[0]
    print np.mean(accuracy)

data = np.genfromtxt("train.txt",delimiter=",",usecols=range(1 , 10))
labels = np.genfromtxt("train.txt",delimiter=",",usecols=range(10 , 11))

label_to_features = {}
label_to_std = {}
label_to_mean = {}
label_to_prior = {}
label_to_pdf_to_feature = {}

for label,features in zip(labels,data):
    if (label not in label_to_features):
        label_to_features[label] = []
    
    label_to_features[label].append(features)

for label in label_to_features:
    label_to_mean[label] = np.mean(label_to_features[label], axis=0)
    label_to_std[label] = np.std(label_to_features[label], axis=0)
    # 9 features
    label_to_pdf_to_feature[label] = range(0 , 9)
    for i in range(0 , 9):
        label_to_pdf_to_feature[label][i] = scipy.stats.norm(label_to_mean[label][i], label_to_std[label][i])
        
    label_to_prior[label] = ( len(label_to_features[label]) * 1.0 ) / len(labels)

print "Naive Bayes"
print "Test - "
classify("test.txt", label_to_mean, label_to_std, label_to_pdf_to_feature, labels, label_to_prior)

print "Training - "
classify("train.txt", label_to_mean, label_to_std, label_to_pdf_to_feature, labels, label_to_prior)

