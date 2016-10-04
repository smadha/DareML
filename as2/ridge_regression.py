'''
Uses data from data_analysis and runs linear regression on it converging through gradient descent
'''
import numpy as np
from data_analysis import get_norm_data
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter

def calc_squared_error(y_pred,y_true):
    '''
    Calculate mean squared error between true and predicted values 
    '''
    return (y_pred - y_true) * (y_pred - y_true)  
    
def pred_value(x_i, weights):
    '''
    Outputs value of x_i using weights. bias = weights[0], x_i[0] = 1
    f(x_i,w,b) = weight * x_i
    '''
    return np.dot(weights, x_i)
    
    
def calc_weights(X, Y, lamb):
    '''
    X -> 2D array of features. Num of samples X (Num of features )
    We add X[:,0] -> Bias
    weights = (X^T X + lambda*I)^-1 * X^T * Y
    return weight
    '''
    lambda_iden = np.identity(len(X[0])) * lamb
    X_t = np.transpose(X)
    X_t_X_inv = np.linalg.pinv( np.dot(X_t,X) + lambda_iden ) 
    X_t_X_inv_X_t = np.dot(X_t_X_inv, X_t)
    
    return np.dot(X_t_X_inv_X_t, Y)   

def calcl_SE(data, label, weights):
    squared_error = 0
    for i,x_i in enumerate(data):
        label_pred = pred_value(x_i, weights)
#         print label_pred, label[i]
        squared_error+=calc_squared_error(label_pred,label[i])
    
    return squared_error

def shuffle_in_unison_inplace(a, b):
    '''
    shuffle a and b such that pir<a_i,b_i> will always exist in new array
    '''
    perm = np.random.permutation(len(a))
    return a[perm], b[perm]


print "Ridge regression with all normalised features"
  
tr_data,test_data,tr_label,test_label = get_norm_data() 

# add one more column in tr_data for bias
tr_data = np.insert(tr_data, [0],1, axis=1)
# add one more column in test_data for bias
test_data = np.insert(test_data, [0],1, axis=1)

# Try lambda values 0.01, 0.1, 1.0
for lamb in [0.01, 0.1, 1.0]:
    weights = calc_weights(tr_data, tr_label, lamb)

    print "Ridge lambda = " + str(lamb) + " MSE TEST = " + str(calcl_SE(test_data, test_label, weights)/len(test_data))
    print "Ridge lambda = " + str(lamb) + " MSE TRAINING = " + str(calcl_SE(tr_data, tr_label, weights)/len(tr_data))

# Use CV to find best lambda  [0.0001, 10]
lamb_to_error = Counter({})

# split it into 10 folds
fold_size = int(len(tr_data)/10)
remainder_array = [1,2,3,3,3,3,3,3,3]

folds = np.split(tr_data,[(i+1)*x + (remainder_array[i]) for i,x in enumerate([fold_size]*9) ])

split_idx = [0] + [(i+1)*x + (remainder_array[i]) for i,x in enumerate([fold_size]*9) ] + [len(tr_data)]

tr_data,tr_label = shuffle_in_unison_inplace(tr_data,tr_label)
for lamb in  [x * 0.0001 for x in range(1, 100000,1000)]:
    
    mse_ev = 0

    for i in range(0,len(folds) ):

        tr_idx = [1]*len(tr_data)
        for sel in range(split_idx[i],split_idx[i+1]):
            tr_idx[sel] = 0
        
        ev_idx = [0]*len(tr_data)
        for sel in range(split_idx[i],split_idx[i+1]):
            ev_idx[sel] = 1

        tr_fold, ev_fold = np.array(list(itertools.compress(tr_data, tr_idx))), np.array(list(itertools.compress(tr_data, ev_idx)))
        tr_fold_label, ev_fold_label = np.array(list(itertools.compress(tr_label, tr_idx))), np.array(list(itertools.compress(tr_label, ev_idx)))
        
        
        # Train on 9 fold
        weights = calc_weights(tr_fold, tr_fold_label, lamb)
        
        # Test on 1 fold
        mse_ev += calcl_SE(ev_fold, ev_fold_label, weights)/len(ev_fold)
    
    lamb_to_error[lamb] = mse_ev / len(folds)


best_lambda_error = lamb_to_error.most_common()[-1:][0]
print "\nBest lambda found by cross validation= ", best_lambda_error[0], "MSE for CV = ",best_lambda_error[1]    

weights = calc_weights(tr_data, tr_label, best_lambda_error[0])
print "Using this lambda = " + str(best_lambda_error[0]) + " MSE TEST = " + str(calcl_SE(test_data, test_label, weights)/len(test_data))

x,y=[],[]
for val in sorted(lamb_to_error.items(), key=itemgetter(0)):
    x.append(val[0])
    y.append(val[1])
    
plt.plot(x,y)
plt.savefig("Ridge_CV_RESULTS.png")