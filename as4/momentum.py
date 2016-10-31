'''
Read about momentum for Stochastic Gradient Descent. We will
use a variant of basic momentum techniques called the Nesterov momentum. Train the
same architecture as in the previous part (with ReLu hidden activations and softmax final
activation) with the following parameters: regularization coefficient = 0.0, number of epochs
= 50, batch size = 1000, learning rate = 10-5
, decay = best value found in last part, Nesterov
= True, Early Stopping = False and a list of momentum coefficients = [0.99, 0.98, 0.95, 0.9,0.85]. 
Find the best value for the momentum coefficients, which gives the maximum test set
accuracy
'''
from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 

print "Momentum"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 800, 500, 300, dout]]
l2_reg_coeffs = [0.0]
decays = [5e-5]
moms = [0.99, 0.98, 0.95, 0.9,0.85]

testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 50, batch_size = 1000, sgd_lr = 1e-5, sgd_decays = decays, sgd_moms = moms,
                    sgd_Nesterov = True, EStop = False, verbose = 0)

print "done"
