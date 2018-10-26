# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:57:47 2017

@author: Administrator
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import getPath
from sklearn.metrics import accuracy_score

ntrain_high=2262
ntrain_low=6581
ntest_high=2262
ntest_low=6580
ntrain = 8843
ntest = 8842
count = 17685
feature_vim=7

feature_train=np.zeros((feature_vim,ntrain))
root_train='E:/featureData_CUHK/train'
paths_train,count_train=getPath.getPath(root_train)
i=0
for path in paths_train:
    feature=np.load(path) 
    feature_train[i]=feature
    i=i+1
train_feature=np.transpose(feature_train)  

label_train=np.array([])
for i in range(1,ntrain_high+1):
    label_train=np.append(label_train,1) 
for j in range(1,ntrain_low+1):
    label_train=np.append(label_train,0)
train_label=np.transpose(label_train)

#crange=[-5,-3,-1,1,3,5,7,9,11,13,15]
#train the svm
clf = svm.SVR()
clf.fit(train_feature,train_label)

feature_test = np.zeros((feature_vim,ntest))
root_test='E:/featureData_CUHK/test'
paths_test,count_test=getPath.getPath(root_test)
i=0
for path in paths_test:
    feature=np.load(path)  
    feature_test[i]=feature
    i=i+1
test_feature=np.transpose(feature_test)  

label_test=np.array([])
for i in range(1,ntest_high+1):
    label_test=np.append(label_test,1)  
for j in range(1,ntest_low+1):
    label_test=np.append(label_test,0)
test_label=np.transpose(label_test)

#test on testset
predict_label = clf.fit(train_feature,train_label).decision_function(test_feature)
predict_label1=clf.predict(test_feature)
print predict_label
print predict_label1
fpr,tpr,thresholds  = roc_curve(test_label,predict_label)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=2 #line width
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

score = clf.score(test_feature,test_label)
print score

print test_label
print predict_label
print np.sum(predict_label)
print np.sum(predict_label1)
#下面是计算classification accurate
#the fraction of correctly classified samples
accuracy=accuracy_score(test_label, predict_label1)
print accuracy
#the number of correctly classified samples
correctly_classified=accuracy_score(test_label, predict_label1, normalize=False)
print correctly_classified