# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:28:39 2017

@author: Administrator
"""

import numpy as np
import getPath

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

np.save('E:/featureData_CUHK/trainfeature.npy',train_feature)
np.save('E:/featureData_CUHK/trainlabel.npy',train_label)

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

np.save('E:/featureData_CUHK/testfeature.npy',test_feature)
np.save('E:/featureData_CUHK/testlabel.npy',test_label)