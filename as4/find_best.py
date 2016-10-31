import theano
theano.config.openmp = True
OMP_NUM_THREADS=8 

import linear_activations
import sigmoid_activations
import relu_activations
import l2_regularization
import early_l2_regularization
import SGD_weight_decay
import momentum 