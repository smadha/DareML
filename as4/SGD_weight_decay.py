'''
SGD with weight decay: (5 Points) During gradient descent, it is often a good idea to
start with a big value of the learning rate and then reduce it as the number of iterations
progress 
In this part we will experiment with the decay factor. Use the network [din, 800, 500,
300, dout]; all hidden activations ReLu and output activation softmax. Use a regularization
coefficient = 5 x 10-7, number of epochs = 100, batch size = 1000, learning rate = 10-5,
and a list of decays: [1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]. Use no momentum and
no early stopping. Report your test set accuracies for the decay parameters and choose the
best one based on your observations.
'''
from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 

print "SGD with weight decay"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 800, 500, 300, dout]]
l2_reg_coeffs = [5e-7]
decays = [1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]

testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 100, batch_size = 1000, sgd_lr = 1e-5, sgd_decays = decays, sgd_moms = [0.0],
                    sgd_Nesterov = False, EStop = False, verbose = 0)

print "done"
