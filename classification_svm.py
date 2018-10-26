# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 22:18:20 2016

@author: Administrator
"""
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,accuracy_score

feature_vim=7
train_feature=np.load('E:/featureData_CUHK/trainfeature.npy')
train_label=np.load('E:/featureData_CUHK/trainlabel.npy')
test_feature=np.load('E:/featureData_CUHK/testfeature.npy')
test_label=np.load('E:/featureData_CUHK/testlabel.npy')

#crange=[-5,-3,-1,1,3,5,7,9,11,13,15]
clf = svm.LinearSVC(C=2**-3)

#clf = svm.SVC()
clf.fit(train_feature,train_label)

predict_label = clf.decision_function(test_feature)
np.save('C:/Users/Administrator/Desktop/baseline_liner/score_cuhkpq_ke.npy',predict_label)

predict_label1=clf.predict(test_feature)
print predict_label
print predict_label1
fpr,tpr,thresholds  = roc_curve(test_label,predict_label)
roc_auc = auc(fpr,tpr)
#np.save('E:/color_CUHK/data/predict_CUHK_color2.npy',predict_label)
#

plt.figure()
lw=2 
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

score = clf.score(test_feature,test_label)
print 'Accuracy: %0.4f' % score
"""
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
print correctly_classified"""