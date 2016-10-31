'''
Next let us try sigmoid activations. We will only explore
the bigger architectures though. Train models of the following architectures: [din, 50, dout],
[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout];
all hidden layers with sigmoids and output layer with softmax. Keep all other parameters
the same as with linear activations. Report your test set accuracies and comment on the
trend of accuracies obtained with changing model architectures. Also explain why this trend
is different from that of linear activations. Report and compare the time taken to train these
architectures with those for linear architectures.
'''
from hw_utils import testmodels,X_tr,y_tr,X_te,y_te 
import time

print "Sigmoid activation"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [[din, 50, dout],[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]

start_time = time.time()
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()

print "Sigmoid activations ends in {0:.3f} seconds".format(end_time-start_time)

