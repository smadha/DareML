'''
Uses results of previous analysis
'''
from hw_utils import testmodels, X_tr, y_tr, X_te, y_te 
import time

print "Combining previous analysis"
din, dout = len(X_tr[0]), len(y_tr[0])

arch = [ [din, 800, 500, 300, dout]]
l2_reg_coeffs = [5e-07]
decays = [5e-05]
moms = [0.90]

start_time = time.time()
testmodels(X_tr, y_tr, X_te, y_te, arch, actfn='relu', last_act='softmax', reg_coeffs = l2_reg_coeffs,
                num_epoch = 50, batch_size = 1000, sgd_lr = 1e-5, sgd_decays = decays, sgd_moms = moms,
                    sgd_Nesterov = True, EStop = False, verbose = 0)
end_time = time.time()

print "Combining previous analysis ends in {0:.3f} seconds".format(end_time-start_time)
