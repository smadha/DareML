'''
Cross Validation Accuracy = 97.05% Above for C= 4^0 gamma= 4^-1 time=0.81233382225:
'''
from scipy.io import loadmat
import numpy as np
from svmutil import *
import time
from plot_3d import *
from libsvm_analyse import prob, TEST_DATA, TEST_LABEL

print "Running RBF kernel for C= 4^0 gamma= 4^-1"

options = svm_parameter('-s 0 -t 2 -c {0} -g {1} -q'.format(1,4**-1))
model = svm_train(prob,options)
    #     svm_train(TRAINING_LABEL.tolist()[0], TRAINING_DATA.tolist(),'-c {0} -v 3 -q'.format(4**i))

svm_predict(TEST_LABEL[0].tolist(), TEST_DATA.tolist(), model)