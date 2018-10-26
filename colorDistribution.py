# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 09:10:16 2016

@author: Administrator
"""

import numpy as np
import cv2

def extractfeature(indices,counts,rootdir):
    data=np.zeros((counts,4096))
    for i in range(counts):
        path=rootdir+indices[i]+'.jpg'
        print 'extracting '+path
        img=cv2.imread(path)
        height, width = img.shape[:2] 
        (B,G,R)=cv2.split(img)    
        pixel=np.zeros((1,4096))   
        for x in range(height):
            for y in range(width):
                r=R[x][y]
                g=G[x][y]
                b=B[x][y]
                r=r/16
                g=g/16
                b=b/16
                index=r*16*16+g*16+b
                pixel[0][index]=pixel[0][index]+1
       
        pixel=pixel/(width*height)
        np.save(rootdir+'/{0}.npy'.format(i),pixel)
        pixel=np.array(pixel)
        data[i-1]=pixel      
        i=i+1 
    return data

def classify(root,counts,dataSet,labels):
    print 'classify'
    q_color=np.array([])
    k=5  
    for i in range(1,counts+1):
        path=root+'/{0}.npy'.format(i)
        print 'processing '+path
        testsample=np.load(path)    
        dataSetSize = dataSet.shape[0]
        diffMat=np.tile(testsample,(dataSetSize,1))- dataSet
        absDiffMat =abs(diffMat) 
        distances =absDiffMat.sum(axis=1) 
        sortedDistIndicies = distances.argsort()  
        npr=0
        nsn=0
        for j in range(k):
            vote = labels[sortedDistIndicies[j]]
            if vote==0:
                nsn=nsn+1
            if vote==1:
                npr=npr+1
        qcd=npr-nsn
        q_color=np.append(q_color,qcd)
    return q_color
    
if __name__=='__main__':
    rootDir="E:/AVA/"   
    n_train=46532
    n_test=11633
    
    train_indices=np.load('E:/ava_delta1/train/train_indices.npy')
    test_indices=np.load('E:/ava_delta1/test/test_indices.npy')

    dataSet_train=extractfeature(train_indices,n_train,rootDir)   
    dataSet_test=extractfeature(test_indices,n_test,rootDir)  
    
    labels=[] 
    for j in range(2262):
        labels.append(1)
    for k in range(6581):
        labels.append(0)
    
    dataSet_train=CreateDataSet(rootdir1,countsTrain)
    np.save('E:/color_CUHK/data/colortest/dataSet_train.npy',dataSet_train)
    print 'train finished'
    dataSet_test=CreateDataSet(rootdir2,countsTest)
    np.save('E:/color_CUHK/data/colortest/dataSet_test.npy',dataSet_test)
    print 'test finished'
    
    """
    qcolor_train=classify(rootdir1,countsTrain,dataSet_train,labels)  
    np.save('E:/featureData_CUHK/train/qcolor_train.npy',qcolor_train)
    print 'train classified finished'
    qcolor_test=classify(rootdir2,countsTest,dataSet_train,labels)
    np.save('E:/featureData_CUHK/test/qcolor_test.npy',qcolor_test)
    print 'test classified fdinished'
    return qcolor_train,qcolor_test
    """
