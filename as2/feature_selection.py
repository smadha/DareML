'''
LSTAT    -0.7399698206    0.74    12
RM    0.690923335    0.69    5
PTRATIO    -0.5052707569    0.51    10
INDUS    -0.4830674218    0.48    2
'''

import numpy as np
from data_analysis import get_norm_data, boston
from linear_regression import report_linear_regression

best_4 = [12,5,10,2]
print "Selected 4 features with highes co-relation ", boston.feature_names[best_4] 

print "Linear regression with these 4 features"
tr_data,test_data,tr_label,test_label = get_norm_data()  

report_linear_regression(tr_data[:,best_4 ],test_data[:,best_4 ],tr_label,test_label)

