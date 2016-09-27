'''
Uses data from data_analysis and runs linear regression on it converging through gradient descent
'''
import numpy as np
from data_analysis import get_norm_data

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
    
    
def calc_weights(X, Y):
    '''
    X -> 2D array of features. Num of samples X (Num of features )
    We add X[:,0] -> Bias
    weights = (X^T X)^-1 * X^T * Y
    return weight
    '''

    X_t = np.transpose(X)
    X_t_X_inv = np.linalg.inv( np.dot(X_t,X))
    X_t_X_inv_X_t = np.dot(X_t_X_inv, X_t)
    
    return np.dot(X_t_X_inv_X_t, Y)   
    
def calcl_MSE(data, label, weights):
    squared_error = 0
    for i,x_i in enumerate(data):
        label_pred = pred_value(x_i, weights)
#     print label_pred, test_label[i]
        squared_error+=calc_squared_error(label_pred,label[i])
    
    return squared_error

def get_weights(tr_data, tr_label):
    '''
    Get optimised weights
    '''
    # add one more column in tr_data for bias
    tr_data = np.insert(tr_data, [0],1, axis=1)
    
    weights = calc_weights(tr_data, tr_label)
    
    return weights, tr_data

def calcl_residual(data, label):
    '''
    Return y_pred - y_true
    '''
    weights,data = get_weights(data, label)
    residual = []
    for i,x_i in enumerate(data):
        label_pred = pred_value(x_i, weights)
#         print label_pred, label[i]
        residual.append(label_pred - label[i])
    
    return np.array(residual)
    
def report_linear_regression(tr_data,test_data,tr_label,test_label, console=True):
    
    weights, tr_data = get_weights(tr_data, tr_label)
    
    # add one more column in test_data for bias
    test_data = np.insert(test_data, [0],1, axis=1)
    mse_test = calcl_MSE(test_data, test_label, weights)/len(test_data)
    mse_train = calcl_MSE(tr_data, tr_label, weights)/len(tr_data) 
    if console:
        print "MSE TEST = " + str(mse_test)
        print "MSE TRAINING = " + str(mse_train)
    
    return mse_test, mse_train
        
        
def main_fn():
    print "Linear regression with all normalised features"

    tr_data,test_data,tr_label,test_label = get_norm_data() 
    
    report_linear_regression(tr_data,test_data,tr_label,test_label)



if __name__ == '__main__':
    main_fn()