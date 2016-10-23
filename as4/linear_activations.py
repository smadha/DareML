'''
First we will explore networks with linear activations. Train
models of the following architectures: [din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50,
50, 50, dout] each having linear activations for all hidden layers and softmax activation for
the last layer. Use 0.0 regularization parameter, set the number of epochs to 30, batch size
to 1000, learning rate to 0.001, decay to 0.0, momentum to 0.0, Nesterov flag to False, and
Early Stopping to False. Report the test set accuracies and comment on the pattern of test
set accuracies obtained. Next, keeping the other parameters same, train on the following
architectures: [din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout],
[din, 800, 800, 500, 300, dout]. Report the observations and explain the pattern of test set
accuracies obtained. Also report the time taken to train these new set of architectures.
'''
from hw_utils import testmodels,X_tr,y_tr,X_te,y_te 

din, dout = len(X_tr[0]), len(y_tr[0])

arch = [[din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50, 50, 50, dout]]
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=False, verbose=0)

arch = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout]]
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=False, verbose=0)

print "done"
