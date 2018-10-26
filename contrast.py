# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 13:09:16 2016

@author: Administrator
"""
import cv2
import numpy as np
import getPath

print 'begin'
def contrast(paths):
    q_contrast=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        height, width = img.shape[:2] 
        histb=cv2.calcHist([img],[0],None,[256],[0.0,256.0])
        histg=cv2.calcHist([img],[1],None,[256],[0.0,256.0])
        histr=cv2.calcHist([img],[2],None,[256],[0.0,256.0])
        hist=histb+histg+histr
        flag= 0.01*3*height*width
        sum1=0
        sum2=0
        for i in range(256):
            sum1=sum1+hist[i]
            if sum1>=flag:
                left=i
                break
        for j in range(256):
            sum2=sum2+hist[256-j-1]
            if sum2>=flag:
                right=256-j-1
                break
        qct=right-left
        q_contrast=np.append(q_contrast,qct)
    return q_contrast
    
root_train = 'E:/ImageDataset/train'  
root_test = 'E:/ImageDataset/test'  
paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)

qcontrast_train=contrast(paths_train)    #对比度
qcontrast_test=contrast(paths_test) 
np.save('E:/featureData/train/qcontrast_train.npy',qcontrast_train)
np.save('E:/featureData/test/qcontrast_test.npy',qcontrast_test)