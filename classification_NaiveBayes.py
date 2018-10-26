# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:40:47 2016

@author: Administrator
"""
import numpy as np
import getPath
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Here,we use GaussianNB
ntrain_high=2262
ntrain_low=6581
ntest_high=2262
ntest_low=6580
ntrain = 8843
ntest = 8842
count = 17685
feature_vim=7

feature_train=np.zeros((feature_vim,ntrain))
root_train='E:/featureData/train/'
paths_train,count_train=getPath.getPath(root_train)
i=0
for path in paths_train:
    feature=np.load(path)  
    feature=np.array(feature)
    feature_train[i]=feature
    i=i+1
train_feature=np.transpose(feature_train) 

label_train=np.array([])
for i in range(1,ntrain_high+1):
    label_train=np.append(label_train,1) 
for j in range(1,ntrain_low+1):
    label_train=np.append(label_train,0)
train_label=np.transpose(label_train)

feature_test = np.zeros((feature_vim,ntest))
root_test='E:/featureData/test/'
paths_test,count_test=getPath.getPath(root_test)
i=0
for path in paths_test:
    feature=np.load(path)  
    feature=np.array(feature)
    feature_test[i]=feature
    i=i+1
test_feature=np.transpose(feature_test)  

label_test=np.array([])
for i in range(1,ntest_high+1):
    label_test=np.append(label_test,1)  
for j in range(1,ntest_low+1):
    label_test=np.append(label_test,0)
test_label=np.transpose(label_test)

gnb = GaussianNB()
predict_label= gnb.fit(train_feature, train_label).predict(test_feature)
#print("Number of mislabeled points out of a total %d points : %d" % (train_feature.shape[0],(train_label != predict_label).sum()))

fpr,tpr,thresholds  = roc_curve(test_label,predict_label)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=2 #line width
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

score = gnb.score(test_feature,test_label)
#下面是计算classification accurate
#the fraction of correctly classified samples
print accuracy_score(test_label, predict_label)
#the number of correctly classified samples
print accuracy_score(test_label, predict_label, normalize=False)
