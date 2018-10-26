# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:24:02 2016

@author: Administrator
"""
import numpy as np
import getPath
import space
"""import boundingbox
import colorDistribution
import hueCount
import blur
import contrast
import bright"""

root1='E:/ImageDataset_CUHK/test'
root2='E:/ImageDataset_AVA/test'
AUHK_test,counts_test=getPath.getPath(root1)
np.save('E:/path/CUHK_test.npy',AUHK_test)
AVA_test,counts_test=getPath.getPath(root2)
np.save('E:/path/AVA_test.npy',AVA_test)

"""
root='E:/ImageDataset_CUHK/test'
root_train = 'E:/ImageDataset/train/'  
root_test = 'E:/ImageDataset/test/'  
paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)
print counts_train
print counts_test"""
"""
qedge_train=space.calculateQuality(paths_train)   #边缘空间分布特征
qedge_test=space.calculateQuality(paths_test)
np.save('E:/featureData/train/qedge_train.npy',qedge_train)
np.save('E:/featureData/test/qedge_test.npy',qedge_test)
print 'fdg'

qarea_train=boundingbox.calcArea(paths_train)    #边缘盒面积
qarea_test=boundingbox.calcArea(paths_test)  
np.save('E:/featureData/train/qarea_train.npy',qarea_train)
np.save('E:/featureData/test/qarea_test.npy',qarea_test)

qcolor_train,qcolor_test=colorDistribution.classify()   #颜色分布
np.save('E:/featureData/train/qcolor_train.npy',qcolor_train)
np.save('E:/featureData/test/qcolor_test.npy',qcolor_test)


print 'fdf'
qhue_train=hueCount.hueCount(paths_train)   #颜色数
qhue_test=hueCount.hueCount(paths_test) 
np.save('E:/featureData/train/qhue_train.npy',qhue_train)
np.save('E:/featureData/test/qhue_test.npy',qhue_test)


qblur_train=blur.blur(paths_train)    #模糊
qblur_test=blur.blur(paths_test) 
np.save('E:/featureData/train/qblur_train.npy',qblur_train)
np.save('E:/featureData/test/qblur_test.npy',qblur_test)


qcontrast_train=contrast.contrast(paths_train)    #对比度
qcontrast_test=contrast.contrast(paths_test) 
np.save('E:/test/qcontrast_train.npy',qcontrast_train)
np.save('E:/test/qcontrast_test.npy',qcontrast_test)


qbright_train=bright.bright(paths_train)   #亮度
qbright_test=bright.bright(paths_test)
np.save('E:/featureData/train/qbright_train.npy',qbright_train)
np.save('E:/featureData/test/qbright_test.npy',qbright_test)"""