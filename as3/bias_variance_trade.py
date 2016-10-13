import numpy as np
import matplotlib.pyplot as plt
from linear_regression_as3 import report_linear_regression


def get_feature(x,degree):
    '''
    Returns new feature by applying degree
    '''
    polynomial_feature = []
    for i in range(1,degree+1):
        polynomial_feature.append(x**i)
    
    return np.transpose(polynomial_feature)

def plot_histogram(x, var_name):
    '''
    plots histogram of x array, 10 bins equal size
    '''
    _, hist_data, _ = plt.hist(x, bins=10)
    plt.plot(x=hist_data)
    plt.savefig(var_name, linewidth=0)
    plt.close()

def bias_variance_mse(sample_size):    
    # generate random samples of x and noise
    sample_x = np.random.uniform(-1,1,(100,sample_size))
    noise_x = np.random.normal(0,0.1,(100,sample_size))
    
    # calculate y from random samples
    sample_y = np.zeros((100,sample_size))
    for i in range(len(sample_x)):
        sample_y[i] = np.array([(2*x*x ) for x in sample_x[i]]) + noise_x[i]
    
    # Estimate param and store y_pred, MSE
    g1_y,g1_mse = np.ones((100,sample_size)),np.zeros(100)
    g2_y,g2_mse = np.zeros((100,sample_size)),np.zeros(100)
    g3_y,g3_mse = np.zeros((100,sample_size)),np.zeros(100)
    g4_y,g4_mse = np.zeros((100,sample_size)),np.zeros(100)
    g5_y,g5_mse = np.zeros((100,sample_size)),np.zeros(100)
    g6_y,g6_mse = np.zeros((100,sample_size)),np.zeros(100)
    
    for i in range(len(sample_x)):
        # No features for g1
        g1_mse[i] = np.mean((g1_y[i]-sample_y[i])**2 )
        
        # Constant features for g2
        g2_mse[i],g2_y[i] = report_linear_regression([], sample_y[i], False)
        
        # 1 features for g3
        g3_mse[i],g3_y[i] = report_linear_regression(get_feature(sample_x[i],1), sample_y[i], False)
        
        # 2 features for g4
        g4_mse[i],g4_y[i] = report_linear_regression(get_feature(sample_x[i],2), sample_y[i], False)
        
        # 3 features for g5
        g5_mse[i],g5_y[i] = report_linear_regression(get_feature(sample_x[i],3), sample_y[i], False)
        
        # 4 features for g6
        g6_mse[i],g6_y[i] = report_linear_regression(get_feature(sample_x[i],4), sample_y[i], False)
    
    # print g1_mse[0]
    # plot MSE along with estimated
    plot_histogram(g1_mse, "g1_mse" + str(sample_size) )
    plot_histogram(g2_mse, "g2_mse" + str(sample_size) )
    plot_histogram(g3_mse, "g3_mse" + str(sample_size) )
    plot_histogram(g4_mse, "g4_mse" + str(sample_size) )
    plot_histogram(g5_mse, "g5_mse" + str(sample_size) )
    plot_histogram(g6_mse, "g6_mse" + str(sample_size) )
    
    # Calculate bias using y_pred
    # Bias can also be calculated using MSE = average(MSE)
    bias_sq = [np.mean(g1_mse),np.mean(g2_mse),np.mean(g3_mse),np.mean(g4_mse),np.mean(g5_mse),np.mean(g6_mse)]
    print "BIAS","Sample Size = " + str(sample_size) , bias_sq
    
    # Calculate variance using y_pred, same as np.std()
    variance = [ np.mean((g1_y - np.mean(g1_y))**2),np.mean((g2_y - np.mean(g2_y))**2),np.mean((g3_y - np.mean(g3_y))**2),
                np.mean((g4_y - np.mean(g4_y))**2),np.mean((g5_y - np.mean(g5_y))**2),np.mean((g6_y - np.mean(g6_y))**2) ]
    print "VARIANCE","Sample Size = " + str(sample_size) , variance

def bias_variance_mse_lambda():    
    sample_size=100
    # generate random samples of x and noise
    sample_x = np.random.uniform(-1,1,(100,sample_size))
    noise_x = np.random.normal(0,0.1,(100,sample_size))
    
    # calculate y from random samples
    sample_y = np.zeros((100,sample_size))
    for i in range(len(sample_x)):
        sample_y[i] = np.array([(2*x*x ) for x in sample_x[i]]) + noise_x[i]
    
    # Estimate param and store y_pred, MSE
    lambda_arr = [0.001,0.003,0.01,0.03,0.1,0.3,1.0]
    h_lambda_y,h_lambda_mse = [np.zeros((100,sample_size))]*len(lambda_arr),[np.zeros(100)]*len(lambda_arr)
    
    for i in range(len(sample_x)):
        for idx,lambda_i in enumerate(lambda_arr):
            # 2 features for each h
            h_lambda_mse[idx], h_lambda_y[idx] = report_linear_regression(get_feature(sample_x[i],2), sample_y[i], False,lamb=lambda_i)
        
        
    # plot MSE for each lambda
#     plot_histogram(g1_mse, "g1_mse" + str(sample_size) )
#     plot_histogram(g2_mse, "g2_mse" + str(sample_size) )
#     plot_histogram(g3_mse, "g3_mse" + str(sample_size) )
#     plot_histogram(g4_mse, "g4_mse" + str(sample_size) )
#     plot_histogram(g5_mse, "g5_mse" + str(sample_size) )
#     plot_histogram(g6_mse, "g6_mse" + str(sample_size) )
    
    # Calculate bias using y_pred
    # Bias can also be calculated using MSE = average(MSE)
    bias_sq = []
    for idx,lambda_i in enumerate(lambda_arr):
        bias_sq.append(np.mean(h_lambda_mse[idx]))
        print "BIAS","lambda = " + str(lambda_arr[idx]) , bias_sq[idx]
    
    # Calculate variance using y_pred, same as np.std()
    variance = []
    for idx,lambda_i in enumerate(lambda_arr):
        variance.append(np.std(h_lambda_y[idx])**2)
        print "VARIANCE","lambda = " + str(lambda_arr[idx]) , variance[idx]

# bias_variance_mse(10)
# bias_variance_mse(100)

bias_variance_mse_lambda()
