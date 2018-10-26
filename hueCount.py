# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:03:56 2016

@author: Administrator
"""

import cv2
import numpy as np
import getPath
alp=0.05

def calcHist(img):
    height, width = img.shape[:2] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(hsv) 
    H=H*2.0
    S=S/255.0
    V=V/255.0
    hist=np.zeros((20,1))
    for i in range(height):
        for j in range(width):
            if S[i][j]>0.2 and V[i][j]>=0.15 and V[i][j]<=0.95:
                k=H[i][j]/18
                hist[k][0]=hist[k][0]+1
    return hist

def hueCount(paths):
    q_hue=np.array([])
    for path in paths:
        print 'processing '+path
        image=cv2.imread(path)
        hist=calcHist(image)
        m=hist.max()
        N=sum(hist>alp*m)
        qh=20-N
        q_hue=np.append(q_hue,qh)
    return q_hue
    
root_train = 'E:/ImageDataset/train'  
root_test = 'E:/ImageDataset/test'  
#paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)
#qhue_train=hueCount(paths_train)   
#np.save('E:/featureData/train/qhue_train.npy',qhue_train)
qhue_test=hueCount(paths_test)   
np.save('E:/featureData/test/qhue_test.npy',qhue_test)