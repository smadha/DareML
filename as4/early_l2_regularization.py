'''
Early Stopping and L2-regularization: (5 Points) To prevent overfitting, we will next
apply early stopping techniques. For early stopping, we reserve a portion of our data as a
validation set and if the error starts increasing on it, we stop our training earlier than the
provided number of iterations. We will use 10% of our training data as a validation set
and stop if the error on the validation set goes up consecutively six times. Train the same
architecture as the last part, with the same set of L2-regularization coefficients, but this
time set the Early Stopping flag in the call to testmodels() as True. Again report your
accuracies on the test set and explain the trend of observations. Report the best value of the
regularization hyperparameter this time. Is it the same as with only L2-regularization? Did
early stopping help?'''
from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 

print "Early Stopping and L2-Regularization"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 800, 500, 300, dout]]
l2_reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 30, batch_size = 1000, sgd_lr = 5e-4, sgd_decays = [0.0], sgd_moms = [0.0],
                    sgd_Nesterov = False, EStop = True, verbose = 0)

print "done"
