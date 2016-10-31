'''
Repeat the above part with ReLu activations for the hidden
layers (output layer = softmax). Keep all other parameters and architectures the same, except
change the learning rate to 5 x 10-4. Report your observations and explain the trend again.
Also explain why this trend is different from that of linear activations. Report and compare
the time taken to train these architectures with those for linear and sigmoid architectures
'''
from hw_utils import testmodels,X_tr,y_tr,X_te,y_te 
import time
print "ReLu activation"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [[din, 50, dout],[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]

start_time = time.time()
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()

print "ReLu activations ends in {0:.3f} seconds".format(end_time-start_time)