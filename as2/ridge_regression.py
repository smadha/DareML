'''
Uses data from data_analysis and runs linear regression on it converging through gradient descent
'''
import numpy as np
from data_analysis import get_norm_data
from collections import Counter

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
    X_t_X_inv = np.linalg.inv( np.dot(X_t,X) + lambda_iden ) 
    X_t_X_inv_X_t = np.dot(X_t_X_inv, X_t)
    
    return np.dot(X_t_X_inv_X_t, Y)   

def calcl_MSE(data, label, weights):
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

    print "Ridge lambda = " + str(lamb) + " MSE TEST = " + str(calcl_MSE(test_data, test_label, weights)/len(test_data))
    print "Ridge lambda = " + str(lamb) + " MSE TRAINING = " + str(calcl_MSE(tr_data, tr_label, weights)/len(tr_data))

# Use CV to find best lambda  [0.0001, 10]
lamb_to_error = Counter({})
size_tr_fold = int(0.9 * len(tr_data) )

for lamb in  [x * 0.0001 for x in range(1, 100000,1000)]:
    # shuffle entire training data set
    tr_data,tr_label = shuffle_in_unison_inplace(tr_data,tr_label)
    
    tr_fold, ev_fold = tr_data[:size_tr_fold,:], tr_data[size_tr_fold:,:]
    tr_fold_label, ev_fold_label = tr_label[:size_tr_fold], tr_label[size_tr_fold:]
    
    # Train on 9 fold
    weights = calc_weights(tr_data, tr_label, lamb)
    
    # Test on 1 fold
    lamb_to_error[lamb] = calcl_MSE(tr_data, tr_label, weights)/len(tr_data)



best_lambda_error = lamb_to_error.most_common()[-1:][0]
print "Best lambda", best_lambda_error[0], "MSE- ",best_lambda_error[1]    

weights = calc_weights(tr_data, tr_label, best_lambda_error[0])
print "Using this lambda = " + str(best_lambda_error[0]) + " MSE TEST = " + str(calcl_MSE(test_data, test_label, weights)/len(test_data))

