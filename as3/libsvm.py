from scipy.io import loadmat
import numpy as np
from svmutil import *
import time

''' 
0 having_IP_Address  { -1,1 }
1 URL_Length   { 1,0,-1 }
2 Shortining_Service { 1,-1 }
3 having_At_Symbol   { 1,-1 }
4 double_slash_redirecting { -1,1 }
5 Prefix_Suffix  { -1,1 }
6 having_Sub_Domain  { -1,0,1 }
7 SSLfinal_State  { -1,1,0 }
8 Domain_registeration_length { -1,1 }
9 Favicon { 1,-1 }
10 port { 1,-1 }
11 HTTPS_token { -1,1 }
12 Request_URL  { 1,-1 }
13 URL_of_Anchor { -1,0,1 }
14 Links_in_tags { 1,-1,0 }
15 SFH  { -1,1,0 }
16 Submitting_to_email { -1,1 }
17 Abnormal_URL { -1,1 }
18 Redirect  { 0,1 }
19 on_mouseover  { 1,-1 }
20 RightClick  { 1,-1 }
21 popUpWidnow  { 1,-1 }
22 Iframe { 1,-1 }
23 age_of_domain  { -1,1 }
24 DNSRecord   { -1,1 }
25 web_traffic  { -1,0,1 }
26 Page_Rank { -1,1 }
27 Google_Index { 1,-1 }
28 Links_pointing_to_page { 1,0,-1 }
29 Statistical_report { -1,1 }
30 Result  { -1,1 }
'''

TEST_DATA = loadmat("phishing-test.mat")['features']
TRAINING_DATA = loadmat("phishing-train.mat")['features']
TEST_LABEL = loadmat("phishing-test.mat")['label']
TRAINING_LABEL = loadmat("phishing-train.mat")['label']

def split_multi_val(col, DATA):
    col_values = []
    arr_0, arr_1, arr__1 = [],[],[]
    for val in DATA[:,col]:
        if val == 0:
            arr_0.append(1)
            arr_1.append(0)
            arr__1.append(0)
            
        if val == 1:
            arr_1.append(1)
            arr_0.append(0)
            arr__1.append(0)
            
        if val == -1:
            arr__1.append(1)
            arr_0.append(0)
            arr_1.append(0)
    
    col_values.append(arr_0)
    col_values.append(arr_1)
    col_values.append(arr__1)
    return col_values

col_fixes = np.array([2, 7, 8, 14, 15, 16, 26, 29])-1
col_values_train = []
col_values_test = []

for col in col_fixes:
    col_values_train +=split_multi_val(col, TRAINING_DATA)
    col_values_test +=split_multi_val(col, TEST_DATA)
    


col_added = range(len(TRAINING_DATA[0]),len(TRAINING_DATA[0])+len(col_values_train))

for idx,i in enumerate(col_added):
    col = np.transpose(col_values_train[idx])
    TRAINING_DATA = np.insert(TRAINING_DATA, i, col, axis=1)
    
    col = np.transpose(col_values_test[idx])
    TEST_DATA = np.insert(TEST_DATA, i, col, axis=1)
    

TRAINING_DATA = np.delete(TRAINING_DATA, col_fixes,axis=1)
TEST_DATA = np.delete(TEST_DATA, col_fixes,axis=1)

print "data pre-processed"

prob = svm_problem(TRAINING_LABEL.tolist()[0], TRAINING_DATA.tolist())

print "Cross validation with libsvm"
run_time = []
poly_acc = []
rbf_acc = []
# for i in range(-6,3):
#     start = time.time()
#     param = svm_parameter('-c {0} -v 3 -q'.format(4**i))
#     print "for C= 4^{0} -".format(i) 
#     svm_train(prob,param)
#     run_time.append(time.time()-start)
# 
# print "Average Training time normal SVM- ", np.mean(run_time)
#
# print "Polynomial kernel"
# run_time = []
# for i in range(-3,8):
#     for d in range(1,4):
#         start = time.time()
#         param = svm_parameter('-c {0} -v 3 -t 1 -d {1} -q'.format(4**i,d))
#         print "for C= 4^{0} degree= {1} :".format(i, d) 
#         poly_acc.append([i,d,svm_train(prob,param)])
#         run_time.append(time.time()-start)
# 
# print "Average Training time Polynomial kernel- ", np.mean(run_time)


print "RBF kernel"
run_time = []
for i in range(-3,8):
    for g in range(-7,0):
        start = time.time()
        param = svm_parameter('-c {0} -v 3 -t 2 -g {1} -q'.format(4**i,4**g))
        print "for C= 4^{0} gamma= 4^{1} :".format(i, g) 
        rbf_acc.append([i,g,svm_train(prob,param)])
        run_time.append(time.time()-start)

print "Average Training time RBF kernel- ", np.mean(run_time)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rbf_acc=np.array(rbf_acc)
print rbf_acc, max(rbf_acc[:,2])

ax.plot_wireframe(rbf_acc[:,0], rbf_acc[:,1], rbf_acc[:,2],rstride=10, cstride=10, label='RBF accuracy')
ax.legend()

plt.show()


