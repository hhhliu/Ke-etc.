# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 21:22:59 2016

@author: Administrator
"""

import numpy as np 
import cv2
import getPath
def blur(paths):
    q_blur=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path, 0)
        height,width = img.shape[:2] 
        blur = cv2.GaussianBlur(img,(3,3),0) 
        f = np.fft.fft2(blur) 
        fshift = np.fft.fftshift(f) 
        
        fimg = np.log(np.abs(fshift)) 

        colC=sum(fimg>5)
        C=sum(colC)

        Ib=height*width
        Ib=float(Ib)
        qf=C/Ib
        q_blur=np.append(q_blur,qf)
    return q_blur
    
#root_train = 'E:/ImageDataset/train'  
root_test = 'E:/ImageDataset_CUHK/test/'  
#paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)
print paths_test
#qblur_train=blur(paths_train)   
#np.save('E:/featureData/train/qblur_train.npy',qblur_train)
#qblur_test=blur(paths_test)   
#np.save('E:/featureData/test/qblur_test.npy',qblur_test)