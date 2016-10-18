import numpy as np
import matplotlib.pyplot as plt
from linear_regression_as3 import report_linear_regression

TEST_SIZE = 100
test_x = np.random.uniform(-1, 1, TEST_SIZE)
test_y = np.array([(2 * x * x) for x in test_x]) + np.random.normal(0, 0.1)


def draw_line(x, y, fig_name,xlabel="X",ylabel="Y"):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.figtext(.02, .02, fig_name)
    plt.savefig(fig_name)
    plt.close()
    
def get_feature(x, degree):
    '''
    Returns new feature by applying degree
    '''
    polynomial_feature = []
    for i in range(1, degree + 1):
        polynomial_feature.append(x ** i)
    
    return np.transpose(polynomial_feature)

def plot_histogram(x, var_name):
    '''
    plots histogram of x array, 10 bins equal size
    '''
    _, hist_data, _ = plt.hist(x, bins=10)
    plt.plot(x=hist_data)
#     plt.savefig(var_name, linewidth=0)
    plt.close()

def bias_variance_mse(sample_size):    
    # generate random samples of x and noise
    sample_x = np.random.uniform(-1, 1, (100, sample_size))
    noise_x = np.random.normal(0, 0.1, (100, sample_size))
    
    # calculate y from random samples
    sample_y = np.zeros((100, sample_size))
    for i in range(len(sample_x)):
        sample_y[i] = np.array([(2 * x * x) for x in sample_x[i]]) + noise_x[i]
    
    
    # Estimate param and store y_pred, MSE
    g1_y, g1_mse,g1_test_y, g1_test_mse = np.ones((100, sample_size)), np.zeros(100),np.ones((100, TEST_SIZE)), np.zeros(100)
    g2_y, g2_mse,g2_test_y, g2_test_mse = np.zeros((100, sample_size)), np.zeros(100),np.zeros((100, TEST_SIZE)), np.zeros(100)
    g3_y, g3_mse,g3_test_y, g3_test_mse = np.zeros((100, sample_size)), np.zeros(100),np.zeros((100, TEST_SIZE)), np.zeros(100)
    g4_y, g4_mse,g4_test_y, g4_test_mse = np.zeros((100, sample_size)), np.zeros(100),np.zeros((100, TEST_SIZE)), np.zeros(100)
    g5_y, g5_mse,g5_test_y, g5_test_mse = np.zeros((100, sample_size)), np.zeros(100),np.zeros((100, TEST_SIZE)), np.zeros(100)
    g6_y, g6_mse,g6_test_y, g6_test_mse = np.zeros((100, sample_size)), np.zeros(100),np.zeros((100, TEST_SIZE)), np.zeros(100)
    
    for i in range(len(sample_x)):
        # No features for g1
        g1_mse[i], g1_test_mse[i] = np.mean((g1_y[i] - sample_y[i]) ** 2), np.mean((g1_test_y[i] - test_y) ** 2)
        
        # Constant features for g2
        g2_mse[i], g2_y[i], g2_test_mse[i], g2_test_y[i] = report_linear_regression([], sample_y[i], False, lamb=0, test_data=[], test_label=test_y)
        
        # 1 features for g3
        g3_mse[i], g3_y[i], g3_test_mse[i], g3_test_y[i] = report_linear_regression(get_feature(sample_x[i], 1), sample_y[i], False, lamb=0, test_data=get_feature(test_x,1) , test_label=test_y)
        
        # 2 features for g4
        g4_mse[i], g4_y[i], g4_test_mse[i], g4_test_y[i] = report_linear_regression(get_feature(sample_x[i], 2), sample_y[i], False, lamb=0, test_data=get_feature(test_x,2) , test_label=test_y)
        
        # 3 features for g5
        g5_mse[i], g5_y[i], g5_test_mse[i], g5_test_y[i] = report_linear_regression(get_feature(sample_x[i], 3), sample_y[i], False, lamb=0, test_data=get_feature(test_x,3) , test_label=test_y)
        
        # 4 features for g6
        g6_mse[i], g6_y[i], g6_test_mse[i], g6_test_y[i] = report_linear_regression(get_feature(sample_x[i], 4), sample_y[i], False, lamb=0, test_data=get_feature(test_x,4) , test_label=test_y)
        
        
    # print g1_mse[0]
    # plot MSE along with estimated
    plot_histogram(g1_mse, "g1_mse" + str(sample_size))
    plot_histogram(g2_mse, "g2_mse" + str(sample_size))
    plot_histogram(g3_mse, "g3_mse" + str(sample_size))
    plot_histogram(g4_mse, "g4_mse" + str(sample_size))
    plot_histogram(g5_mse, "g5_mse" + str(sample_size))
    plot_histogram(g6_mse, "g6_mse" + str(sample_size))
    
    # Calculate bias using y_pred
    # Bias can also be calculated using MSE = average(MSE)
#     bias_sq = [np.mean(g1_test_mse), np.mean(g2_test_mse), np.mean(g3_test_mse), np.mean(g4_test_mse), np.mean(g5_test_mse), np.mean(g6_test_mse)]
    bias_sq = [np.mean(g1_mse),np.mean(g2_mse),np.mean(g3_mse),np.mean(g4_mse),np.mean(g5_mse),np.mean(g6_mse)]
    print "BIAS", "Sample Size = " + str(sample_size) 
    print "\n".join(["g{0} - {1}".format(idx + 1, x) for idx, x in enumerate(bias_sq)])
    
    # Calculate variance using y_pred, same as np.std()
    variance = [ np.mean((g1_test_y - np.mean(g1_test_y)) ** 2), np.mean((g2_test_y - np.mean(g2_test_y)) ** 2), np.mean((g3_test_y - np.mean(g3_test_y)) ** 2),
                np.mean((g4_test_y - np.mean(g4_test_y)) ** 2), np.mean((g5_test_y - np.mean(g5_test_y)) ** 2), np.mean((g6_test_y - np.mean(g6_test_y)) ** 2) ]
    print "VARIANCE", "Sample Size = " + str(sample_size) 
    print "\n".join(["g{0} - {1}".format(idx + 1, x) for idx, x in enumerate(variance)])
    
    draw_line(range(6), variance, "Complexity-Variance-Sample-{0}".format(sample_size),"Complexity","Variance")
    draw_line(range(6), bias_sq, "Complexity-Bias-Sample-{0}".format(sample_size),"Complexity","Bias")
    
def bias_variance_mse_lambda():    
    sample_size = 100
    # generate random samples of x and noise
    sample_x = np.random.uniform(-1, 1, (100, sample_size))
    noise_x = np.random.normal(0, 0.1, (100, sample_size))
    
    # calculate y from random samples
    sample_y = np.zeros((100, sample_size))
    for i in range(len(sample_x)):
        sample_y[i] = np.array([(2 * x * x) for x in sample_x[i]]) + noise_x[i]
    
    # Estimate param and store y_pred, MSE
    lambda_arr = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
#     h_lambda_y,h_lambda_mse = [np.zeros((100,sample_size))]*len(lambda_arr),[np.zeros(100)]*len(lambda_arr)
    h_lambda_y, h_lambda_mse,h_lambda_test_y, h_lambda_test_mse  = np.zeros((len(lambda_arr), 100, sample_size)), np.zeros((len(lambda_arr), 100)), np.zeros((len(lambda_arr),100, TEST_SIZE)), np.zeros((len(lambda_arr),100))


    for i in range(len(sample_x)):
        for idx, lambda_i in enumerate(lambda_arr):
            # 2 features for each h
            h_lambda_mse[idx][i], h_lambda_y[idx][i], h_lambda_test_mse[idx][i], h_lambda_test_y[idx][i] = report_linear_regression(get_feature(sample_x[i], 2), sample_y[i], False, lamb=lambda_i, test_data=get_feature(test_x,2) , test_label=test_y)
        
        
    # plot MSE for each lambda
    for idx, lambda_i in enumerate(lambda_arr):
        plot_histogram(h_lambda_mse[idx], "lambda_mse_" + str(lambda_i).replace(".", "_"))
    
    
    # Calculate bias using y_pred
    # Bias can also be calculated using MSE = average(MSE)
    bias_sq = []
    for idx, lambda_i in enumerate(lambda_arr):
        bias_sq.append(np.mean(h_lambda_mse[idx]))
        print "BIAS", "lambda = " + str(lambda_arr[idx]) ,"=", bias_sq[idx]
    
    # Calculate variance using y_pred, same as np.std()
    variance = []
    for idx, lambda_i in enumerate(lambda_arr):
        variance.append(np.std(h_lambda_test_y[idx]) ** 2)
        print "VARIANCE", "lambda = " + str(lambda_arr[idx]) ,"=", variance[idx]
    
    draw_line(lambda_arr, variance, "lambda-Variance-Sample-{0}".format(sample_size),"lambda", "Variance")
    draw_line(lambda_arr, bias_sq, "lambda-Bias-Sample-{0}".format(sample_size),"lambda", "Bias")
    
bias_variance_mse(10)
bias_variance_mse(100)
print "Now using regularization"
bias_variance_mse_lambda()
