#This is linear classification on cifar-10

Softmax approach and SVM approach are used to classify the datas.

When running it, "softmax_lin" or "svm_lin" should be run firstly, to get the training result,

which will be stored in two "pkl" files.

And then "lin_test" can be run to test the accuracy.

###
Unfortunately, these linear approach didn't get really good result(but better than KNN ovo)

###
res:
softmax: 40.1%
svm: 38.6%
