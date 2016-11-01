'''
Grid search with cross-validation: This time we will do a full fledged search for the best architecture 
and parameter combinations. Train networks with architectures [din, 50, dout], [din, 500, dout], 
[din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]; hidden activations 
ReLu and final activation softmax. For each network use the following parameter values: 
number of epochs = 100 batch size = 1000 learning rate = 10-5, Nesterov = True Early Stopping = True
Momentum coefficient = 0.99 (this is mostly independent of other values, so we can directly use it without 
including it in the hyperparameter search). For the other parameters search the full lists: 
for regularization coefficients = [10-7, 5 x 10-7, 10-6, 5 x 10-6, 10-5], and 
for decays = [10-5, 5 x 10-5, 10-4]. 
Report the best parameter values, architecture and the best test set accuracy obtained.
'''

from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 
import time

print "Grid search"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
l2_reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
decays = [1e-5, 5e-5, 1e-4]
moms = [0.99]

start_time = time.time()
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 100, batch_size = 1000, sgd_lr = 1e-5, sgd_decays = decays, sgd_moms = moms,
                    sgd_Nesterov = True, EStop = True, verbose = 0)
end_time = time.time()

print "Grid search {0:.3f} seconds".format(end_time-start_time)
