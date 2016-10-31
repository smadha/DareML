Bias Variance Trade-off (20 Points)
Let the function f (x) = 2x2 + ε, where ε is Gaussian noise drawn from N (0, 0.1). We are also given the following functions:
• g1(x)=1	
• g2(x)=w0	
• g3(x) = w0 + w1x	
• g4(x)=w0 +w1x+w2x2	
• g5(x)=w0 +w1x+w2x2 +w3x3	
• g6(x)=w0 +w1x+w2x2 +w3x3 +w4x4	
￼￼	
(a) Randomly generate 100 datasets. Each dataset contains 10 samples (xi , yi ), where xi is uniformly sampled from [−1,1] and yi = f(xi). For a given g function, estimate its parameters using linear regression (with no regularization) and compute the sum-square-error on every dataset. Then plot the histogram of the mean-squared-error. Repeat this for each g function. Please provide the 6 histogram plots in the report. For each g function, estimate its bias2 and variance, and report the results. (5 points)	
(b) In (a), change the sample size in each dataset to 100, and repeat the procedures. Plot the resulting histograms in the report. For each g function, estimate its bias2 and variance, and report the results. (5 points)	
(c) Given your results in (a) and (b), discuss how the model complexity and sample size affect the bias2 and variance. (5 points)	
(d) Consider function hλ(x) = w0 + w1x + w2x2 where λ ≥ 0 is the L2 regularization parameter used during linear regression i.e. your regression loss function will have an additional λ(w02 + w12 +w2) regularization term. Following (b) (i.e. 100 samples per dataset), estimate the bias2 and variance of each hλ(x) when λ is set to 0.001,0.003,0.01,0.03,0.1,0.3,1.0 respectively. Analyze your obtained values and discuss how λ affects the bias2 and variance. (5 points)	
	
Linear and Kernel SVM (20 Points)		
In this part, you will experiment with SVMs on a real-world dataset. You will implement linear SVM (i.e. SVM using the original features, namely the nonlinear mapping is the identity mapping). Also, you will use a widely used SVM toolbox called LIBSVM to experiment with kernel SVMs.	
Dataset: We have provided the Phishing Websites Data Set from UCI’s machine learning data repository (https://archive.ics.uci.edu/ml/datasets/Phishing+Websites). The dataset is for a bi- nary classification problem to detect phishing websites. The dimensionality of the input feature is 30, and the training and test sets contain 2,000 and 2,000 samples, respectively (they are provided on Blackboard as phishing-train.mat and phishing-test.mat). The datasets can be imported directly in MATLAB and using Scipy.io in Python.	
Data preprocessing	
All the features in the datasets are categorical. You need to preprocess the training and test data to make features with multiple values to features taking values only zero or one. If a feature fi have value {−1, 0, 1}, we create three new features fi,−1 , fi,0 and fi,1 . Only one of them can have value 1 and fi,x = 1 if and only if fi = x. For example, we transform the original feature 1 into [0, 0, 1]. In the given dataset, the features 2, 7, 8, 14, 15, 16, 26, 29 (index starting from 1) take three different values {−1, 0, 1}. You need to transform each above feature into three 0/1 features.	
Use linear SVM in LIBSVM	
LIBSVM is widely used toolbox for SVM and has both a Matlab and a Python interface. Download LIBSVM from https://www.csie.ntu.edu.tw/~cjlin/libsvm/ and install it according to the README file. For each C from {4−6, 4−5, · · · , 4, 42}, apply 3-fold cross validation (use -v option in LIBSVM) and report the cross validation accuracy and average training time. (5 points)	
	
Kernel SVM in LIBSVM	
LIBSVM supports a number of kernel types. Here you need to experiment with the polynomial kernel and RBF (Radial Basis Function) kernel.	
(a) Polynomial kernel. Please tune C and degree in the kernel. For each combination of (C, degree), where C ∈ {4−3, 4−2, · · · , 46, 47} and degree ∈ {1, 2, 3}, report the 3-fold cross validation accuracy and average training time. (5 points)	
(b) RBF kernel. Please tune C and gamma in the kernel. For each combination of (C, gamma), where C ∈ {4−3, 4−2, · · · , 46, 47} and gamma ∈ {4−7, 4−6, · · · , 4−2, 4−1}, report the 3-fold cross validation accuracy and average training time. (5 points)	
Based on the cross validation results of Polynomial and RBF kernel, which kernel type and kernel parameters will you choose? You need to write a script libsvm.m/libsvm.py using the best kernel and the parame- ters selected from cross-validation. Your script should train a SVM using libsvm with selected kernel and parameters and output the accuracy on the test set. (5 points)	
