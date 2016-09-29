'''
Best 4 features 
LSTAT    -0.7399698206    0.74    12
RM    0.690923335    0.69    5
PTRATIO    -0.5052707569    0.51    10
INDUS    -0.4830674218    0.48    2
'''

import numpy as np
from data_analysis import get_norm_data, boston, calc_pearson, norm
from linear_regression import report_linear_regression, calcl_residual
from itertools import permutations

def report_feature_selection(_tr_data,_test_data,_tr_label,_test_label, _best_4):
    print "Selected 4 features- ", boston.feature_names[_best_4] 
    print "Linear regression with these 4 features"  

    report_linear_regression(_tr_data,_test_data,_tr_label,_test_label)
    print ""



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

# Polynomial feature
print "Calculating polynomial features"
num_of_new_features =  ( len(tr_data[0]) * (len(tr_data[0]) + 1)/2 )  + len(tr_data[0])
poly_tr_data, poly_test_data = np.zeros((len(tr_data), num_of_new_features)) , np.zeros((len(test_data), num_of_new_features) )

# copying old features 
for i in range( 0 , len(tr_data[0]) ):
    poly_tr_data[:,i] = tr_data[:,i] 
    poly_test_data[:,i] = test_data[:,i] 

new_col_idx = len(tr_data[0])
for i in range(0,len(tr_data[0])):
    for j in range(i,len(tr_data[0])):
        poly_tr_data[:,new_col_idx] = tr_data[:,i] * tr_data[:,j]
        poly_test_data[:,new_col_idx] = test_data[:,i] * test_data[:,j]
        
        new_col_idx += 1

poly_tr_data, poly_test_data = norm(poly_tr_data, poly_test_data)    

report_linear_regression(poly_tr_data, poly_test_data, tr_label, test_label)
print ""

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







