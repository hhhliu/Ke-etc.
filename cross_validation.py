# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:51:16 2016

@author: Administrator
"""

import numpy as np
from sklearn import svm
from sklearn import cross_validation
#from sklearn.metrics import classification_report
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
import getPath
from sklearn.cross_validation import cross_val_score

ntrain_high=2262
ntrain_low=6581
ntest_high=2262
ntest_low=6580
ntrain = 8843
ntest = 8842
count = 17685
feature_dim=7

feature_train=np.zeros((feature_dim,ntrain))
root_train='E:/featureData_CUHK/train'
paths_train,count_train=getPath.getPath(root_train)
i=0
for path in paths_train:
    feature=np.load(path) 
    feature=np.array(feature)
    feature_train[i]=feature
    i=i+1
train_feature=np.transpose(feature_train)   #求出训练集的特征

label_train=np.array([])
for i in range(1,ntrain_high+1):
    label_train=np.append(label_train,1) 
for j in range(1,ntrain_low+1):
    label_train=np.append(label_train,0)
train_label=np.transpose(label_train)


crange=[-5,-3,-1,1,5,7,9,11,13,15]

for i in crange:
#    tuned_parameters = [{'C': [2**i],'loss': ['hinge']}]
    clf = svm.SVC(kernel='linear',C=2**i) 
    print 'C:%f ' % 2**i
    #kfold=cross_validation.KFold(len(train_feature),n_folds=10)   
    #[clf.fit(train_feature[train], train_label[train]).score(train_feature[test], train_label[test]) for train, test in kfold]              
    scores = cross_val_score(clf,train_feature,train_label,cv=5)
    #The mean score and the 95% confidence interval of the score estimate 
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  