'''
Next we will try to apply regularization to our network. For
this part we will use a deep network with four layers: [din, 800, 500, 300, dout]; all hidden
activations ReLu and output activation softmax. Keeping all other parameters same as for
the previous part, train this network for the following set of L2-regularization parameters:
 [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]. Report your accuracies on the test set and explain the
trend of observations. Report the best value of the regularization hyperparameter.
'''
from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 

print "L2-Regularization"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 800, 500, 300, dout]]
l2_reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 30, batch_size = 1000, sgd_lr = 5e-4, sgd_decays = [0.0], sgd_moms = [0.0],
                    sgd_Nesterov = False, EStop = False, verbose = 0)

print "L2-Regularization ends"
