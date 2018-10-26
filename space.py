# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:06:47 2016

@author: Administrator
"""
import cv2
import numpy as np
import getPath
def extractEdge(img):        
    (B,G,R)=cv2.split(img)    
    gray_lapR = cv2.Laplacian(R,cv2.CV_16S,ksize = 3)  
    dstR = cv2.convertScaleAbs(gray_lapR)          
    gray_lapG = cv2.Laplacian(G,cv2.CV_16S,ksize = 3)  
    dstG = cv2.convertScaleAbs(gray_lapG)        
    gray_lapB = cv2.Laplacian(B,cv2.CV_16S,ksize = 3)  
    dstB = cv2.convertScaleAbs(gray_lapB)      
    
    Meandst=(dstR+dstG+dstB)/3.0 
    resized = cv2.resize(Meandst, (100,100), interpolation=cv2.INTER_AREA)
    sumResized=sum(sum(resized))
    sumResized=float(sumResized)
    resized=resized/sumResized
    return resized
    
def getTrainHighFeature():
    print 'hello'
    rootDir="E:/ImageDataset/train/train_high"
    paths,counts=getPath.getPath(rootDir)
    pro=np.zeros((100,100))
    i=1
    for path in paths:
        print 'trainhigh '+path
        image=cv2.imread(path)
        resize=extractEdge(image)
        np.save('E:/color/data/spatialDistributionofEdges/{0}.npy'.format(i),resize)
        i=i+1     
        pro=pro+resize
    counts=float(counts)
    Mp=pro/counts
    return Mp
    
def getTrainLowFeature():
    rootDir="E:/ImageDataset/train/train_low"
    paths,counts=getPath.getPath(rootDir)
    snap=np.zeros((100,100))
    i=5263
    for path in paths:
        print 'trainlow '+path
        image=cv2.imread(path)
        resize=extractEdge(image)
        np.save('E:/color/data/spatialDistributionofEdges/{0}.npy'.format(i),resize)
        i=i+1     
        snap=snap+resize
    counts=float(counts)
    print counts
    Ms=snap/counts
    return Ms 

def calculateQuality(paths):
    q_edge=np.array([])
    Mp=getTrainHighFeature()
    Ms=getTrainLowFeature()
    for path in paths:
        print 'processing '+path
        image=cv2.imread(path)
        feature=extractEdge(image)
        
        diffdp=feature-Mp
        diffdp=abs(diffdp)
        dp=sum(sum(diffdp))
        
        diffds=feature-Ms
        diffds =abs(diffds)
        ds=sum(sum(diffds))
        
        ql=ds-dp
        q_edge=np.append(q_edge,ql)
    return q_edge

root_train = 'E:/ImageDataset/train'  
root_test = 'E:/ImageDataset/test'  
paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)

qedge_train=calculateQuality(paths_train)   
np.save('E:/featureData/train/qedge_train.npy',qedge_train)
qedge_test=calculateQuality(paths_test)  
np.save('E:/featureData/test/qedge_test.npy',qedge_test)

print qedge_train
print qedge_test