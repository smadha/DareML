'''
Uses data from data_analysis and runs linear regression on it converging through gradient descent
'''
import numpy as np

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
    
    
def calc_weights(X, Y, lamb=0):
    '''
    X -> 2D array of features. Num of samples X (Num of features )
    We add X[:,0] -> Bias
    weights = (X^T X)^-1 * X^T * Y
    return weight
    '''

    lambda_iden = np.identity(len(X[0])) * lamb
    X_t = np.transpose(X)
    X_t_X_inv = np.linalg.pinv( np.dot(X_t,X) + lambda_iden ) 
    X_t_X_inv_X_t = np.dot(X_t_X_inv, X_t)
    
    return np.dot(X_t_X_inv_X_t, Y)   
    
def calcl_MSE(data, label, weights):
    squared_error = 0
    all_label_pred = []
    for i,x_i in enumerate(data):
        label_pred = pred_value(x_i, weights)
#     print label_pred, test_label[i]
        squared_error+=calc_squared_error(label_pred,label[i])
        all_label_pred.append(label_pred)
    
    return squared_error/len(label), np.array(all_label_pred)

def get_weights(tr_data, tr_label,lamb=0):
    '''
    Get optimised weights
    '''
    # add one more column in tr_data for bias
    if tr_data == []:
        tr_data = np.ones(( len(tr_label) ,1))
    else:
        tr_data = np.insert(tr_data, [0],1, axis=1)
    
    weights = calc_weights(tr_data, tr_label, lamb)
    
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
    
def report_linear_regression(tr_data,tr_label, console=True,lamb=0):
    
    weights, tr_data = get_weights(tr_data, tr_label,lamb)
    
    mse_train, all_label_pred = calcl_MSE(tr_data, tr_label, weights) 
    if console:
        print "MSE  = " , str(tr_data[0]) , str(mse_train)
    
    return mse_train, all_label_pred
        
        
