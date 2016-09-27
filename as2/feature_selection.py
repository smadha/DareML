'''
Best 4 features 
LSTAT    -0.7399698206    0.74    12
RM    0.690923335    0.69    5
PTRATIO    -0.5052707569    0.51    10
INDUS    -0.4830674218    0.48    2
'''
import numpy as np
from data_analysis import get_norm_data, boston, calc_pearson
from linear_regression import report_linear_regression, calcl_residual
from itertools import permutations
from collections import Counter

def report_feature_selection(_tr_data,_test_data,_tr_label,_test_label, _best_4):
    print "Selected 4 features- ", boston.feature_names[_best_4] 
    print "Linear regression with these 4 features"  

    report_linear_regression(_tr_data,_test_data,_tr_label,_test_label)
    print ""

def calc_MI(X,Y,bins):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    
    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H


tr_data,test_data,tr_label,test_label = get_norm_data()

# Selection with Correlation
print "Selection with initial corelation"
best_4 = [12,5,10,2]
report_feature_selection(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label, best_4)


# Selection with Correlation using residual
best_4=[12]
for unused in range(0,3):
    residual = calcl_residual(tr_data[:,best_4 ], tr_label)
    max_relation = [-1,-1]
    for i in range(0,len(tr_data[0])):
        if i in best_4: # all features except the chosen ones
            continue 
        
        relation = abs(calc_pearson(tr_data[:,i ], residual))
        if(relation > max_relation[1]):
            max_relation[0] = i
            max_relation[1] = relation
    best_4.append(max_relation[0]) 

print "Selection with iterative residual corelation"
report_feature_selection(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label, best_4)


#Selection by mutual information
best_m_i = Counter({})
for i in range(0,len(tr_data[0])):
    m_i = calc_MI(tr_data[:,i ], tr_label, 10)
    best_m_i[i] = m_i    

print "Highest mutual information", [(boston.feature_names[i[0]],i[1]) for i in best_m_i.most_common(4)]
best_4 = [i[0] for i in best_m_i.most_common(4)]
report_feature_selection(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label, best_4)

# Brute force
print "Selection with brute force"
min_MSE = [[1],9999]
for best_4 in permutations(range(13), 4):
    mse_test, mse_train = report_linear_regression(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label,False)
    
    if(mse_train < min_MSE[1]):
        min_MSE[0] = best_4
        min_MSE[1] = mse_train

best_4 = list(min_MSE[0])
report_feature_selection(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label, best_4)


