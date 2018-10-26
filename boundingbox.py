# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 19:20:26 2016

@author: Administrator
"""
import cv2
import numpy as np

def extractEdge(img):
    (B,R,G)=cv2.split(img)    
    gray_lapR = cv2.Laplacian(R,cv2.CV_16S,ksize = 3)  
    dstR = cv2.convertScaleAbs(gray_lapR)          
    gray_lapG = cv2.Laplacian(G,cv2.CV_16S,ksize = 3)  
    dstG = cv2.convertScaleAbs(gray_lapG)        
    gray_lapB = cv2.Laplacian(B,cv2.CV_16S,ksize = 3)  
    dstB = cv2.convertScaleAbs(gray_lapB)       
    Meandst=(dstR+dstG+dstB)/3.0        
    resized = cv2.resize(Meandst, (100,100), interpolation=cv2.INTER_AREA)
    sumResized=sum(map(sum,resized))
    sumResized=float(sumResized)
    resized=resized/sumResized
    return resized
   
def calcArea(n,indices): 
    rootDir="E:/AVA/" 
    q_area=np.array([])
    for i in range(n):
        path=rootDir+indices[i]+'.jpg'
        print 'processing '+path
        image=cv2.imread(path)
        lapiacian=extractEdge(image)
        px=np.sum(lapiacian,axis=0)
        py=np.sum(lapiacian,axis=1)

        sum1=0
        sum2=0
        flag=0.01
        for i in range(100):
            sum1=sum1+px[i]
            if sum1>=flag:
                leftx=i
                break
        for j in range(100):
            sum2=sum2+px[100-j-1]
            if sum2>=flag:
                rightx=100-j-1
                break
        wx=rightx/100.0-leftx/100.0

        for i in range(100):
            sum1=sum1+py[i]
            if sum1>=flag:
                lefty=i
                break
        for j in range(100):
            sum2=sum2+py[100-j-1]
            if sum2>=flag:
                righty=100-j-1
                break
        wy=righty/100.0-lefty/100.0
        boundingArea=wx*wy
        qa=1-boundingArea
        q_area=np.append(q_area,qa)
    return q_area
    
if __name__=='__main__':
    rootDir="E:/AVA/"   
    n_train=46532
    n_test=11633
    
    train_indices=np.load('E:/ava_delta1/train/train_indices.npy')
    test_indices=np.load('E:/ava_delta1/test/test_indices.npy')
    
    qarea_train=calcArea(n_train,train_indices)    
    np.save('E:/ava_delta1/Ke/data/2qarea_train.npy',qarea_train)
    qarea_test=calcArea(n_test,test_indices)    #边缘盒面积
    np.save('E:/ava_delta1/Ke/data/2qarea_test.npy',qarea_test)
    
    
