'''
Prints histograms of numerical attributes
Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
TARGET  - MEDV     Median value of owner-occupied homes in $1000's
'''
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()

def calc_pearson(x,y):
    return np.corrcoef(x, y)[1, 0]

def plot_histogram(x, var_name):
    # histogram of x data, 10 bins equal size
    _, hist_data, _ = plt.hist(x, bins=10)
    plt.plot(x=hist_data)
    plt.savefig(var_name, linewidth=0)
    plt.close()

def norm(tr_data, test_data):
    '''
    Normamlise data with mean=0 and std_dev=1
    '''
    for i in range(0,len(tr_data[0])):
        v = tr_data[:,i]
        mu =  np.mean(v)
        std_dev = np.std(v)
        tr_data_norm = v - mu
        tr_data_norm = tr_data_norm / std_dev
        tr_data[:,i] = tr_data_norm
        
        v_t = test_data[:,i]
        test_data_norm = v_t - mu
        test_data_norm = test_data_norm / std_dev
        test_data[:,i] = test_data_norm
        
    return tr_data, test_data
    
def get_training_data():
    '''
    all indexes not divisible by 7
    Returns data, target
    '''
    tr_data = []
    target_tr = []
    for i in range(0,len(boston.data)):
        if i % 7 != 0:
            tr_data.append(boston.data[i])
            target_tr.append(boston.target[i])
    
    return np.array(tr_data), np.array(target_tr)
    
def get_norm_data():
    '''
    Normalised trainig and test data. With labels
    returns - [training,test],[training_label,test_label]
    '''
    data_tr,target_tr = get_training_data()
    data_te,target_te = get_test_data()
    data_tr,data_te = norm(data_tr,data_te)
    
    return data_tr,data_te, target_tr,target_te
    
def get_test_data():
    '''
    all indexes divisible by 7
    Returns data, target
    '''
    test_data = []
    target_test = []
    for i in range(0,len(boston.data),7):
        test_data.append(boston.data[i])
        target_test.append(boston.target[i])
    
    return np.array(test_data), np.array(target_test)
    

def main_fn():
    
    # feature_names data DESCR target
        
#     print boston.data[0] , boston.target[0]
#     print boston.data[1] , boston.target[1]

#     print get_test_data()[0][0], get_test_data()[1][0]
#     print get_training_data()[0][0], get_training_data()[1][0]
#     for i in norm_data[0][:,6]: print i #6 age
    tr_data,tr_label = get_training_data()

    # print histogram of all variable    
    for i in range(0, len(boston.data[0])):
        plot_histogram(tr_data[:,i], boston.feature_names[i])

    print "Saved all histograms as image files. Name of file is name of feature"
    
    # print pearsonr correlation of all variable
    print "Pearson correlation of all variables-"
    for i in range(0, len(boston.data[0])):
        print boston.feature_names[i], calc_pearson(tr_data[:,i], tr_label)
    
    tr_data,test_data,tr_label,test_label = get_norm_data()
    print "Normalized Test data example - ", test_data[0],"Label - ", test_label[0]
#     print len(tr_data),len(test_data),len(tr_label),len(test_label) 


    
if __name__ == '__main__':
    main_fn()