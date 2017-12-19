# -*- coding: utf-8 -*-

from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt

#将一个32*32的图像样本转换为1*1024的ndarray
#参数：样本文件名
#返回：1*1024的ndarray
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
    
#欧式距离（Mahalanobis Distance）计算函数    
def EuDistances(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 #each element **2
    sqDistances = sqDiffMat.sum(axis=1) #row sum to one element
    distances = sqDistances ** 0.5
    return distances

#权重计算函数1
def WeightCal1(sortedDistIndicies, distances, i, k):
    weightRange = (float)(distances[sortedDistIndicies[k-1]] - distances[sortedDistIndicies[0]])
    if(weightRange == 0):
        kWeightedDist = 1
    else:
        kWeightedDist = (distances[sortedDistIndicies[k-1]] - distances[sortedDistIndicies[i]])/weightRange
    return kWeightedDist

#权重计算函数2
def WeightCal2(sortedDistIndicies, distances, i, k):
    kWeightedDist = 1.0/distances[sortedDistIndicies[i]]
    return kWeightedDist

#权重计算函数3
def WeightCal3(sortedDistIndicies, distances, i, k):
    kWeightedDist = k-i
    return kWeightedDist

#对输入inX，基于训练样本集(dataSet, labels)用k近邻做出判决, 采用欧氏距离
#返回：inX的判决结果 1*3的resultList
def classify0(inX, dataSet, labels, k):
    resultList = []
    distances = EuDistances(inX, dataSet)    
    sortedDistIndicies = distances.argsort() #return index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        kWeightedDist = WeightCal1(sortedDistIndicies, distances, i, k)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + kWeightedDist #default:0
    sortedClassCount = sorted(classCount.iteritems(), \
                              key=operator.itemgetter(1), reverse=True)
    resultList.append(sortedClassCount[0][0])

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        kWeightedDist = WeightCal2(sortedDistIndicies, distances, i, k)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + kWeightedDist #default:0
    sortedClassCount = sorted(classCount.iteritems(), \
                              key=operator.itemgetter(1), reverse=True)
    resultList.append(sortedClassCount[0][0])

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        kWeightedDist = WeightCal3(sortedDistIndicies, distances, i, k)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + kWeightedDist #default:0
    sortedClassCount = sorted(classCount.iteritems(), \
                              key=operator.itemgetter(1), reverse=True)
    resultList.append(sortedClassCount[0][0])

    return resultList

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)   #trainingDigits文件夹中的训练样本数量
    trainingMat = zeros((m, 1024))#构造一个m*1024的矩阵，用来存储训练样本
    for i in range(m):
        fileNameStr = trainingFileList[i]   #一个样本的文件名
        fileStr = fileNameStr.split('.')[0] #只取文件名，去掉后缀txt
        classNumStr = int(fileStr.split('_')[0])#取出类别
        hwLabels.append(classNumStr)#将Label添加到hwLabels
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) #转化图像数据为1*1024矩阵
    testFileList = listdir('testDigits')
    errorCount = 0.0
    invCovMat = linalg.pinv(cov(trainingMat.T))
    mTest = len(testFileList)


    #正确个数
    correctCntList=[0.0, 0.0, 0.0]
    #数字5识别的TP值
    TP5List= [0.0, 0.0, 0.0]
    #所有判决为5的结果数量 TP+FP
    P5List = [0.01, 0.01, 0.01]
    #数字5识别的TN值
    TN5List = [0.0, 0.0, 0.0]
    #测试样本中数字5的样本数量
    Label5Num = 107.0
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResultList = classify0(vectorUnderTest, \
                                     trainingMat, hwLabels, 3)
        #正确分类的样本数，TP, TN, TP+FP
        for s in range(3):
            if(classifierResultList[s] == classNumStr):
                correctCntList[s] += 1.0
            if(classifierResultList[s] == 5):
                P5List[s] += 1
                if(classNumStr == 5):
                    TP5List[s] += 1
            else:
                if(classNumStr != 5):
                    TN5List[s] += 1
    print("correct rate: %f, %f, %f" %\
              (correctCntList[0]/float(mTest),correctCntList[1]/float(mTest), correctCntList[2]/float(mTest)))
    print("precision rate:%f, %f, %f" %\
               (TP5List[0]/float(P5List[0]),TP5List[1]/float(P5List[1]),TP5List[2]/float(P5List[2])))
    print("recall rate: %f, %f, %f" %\
              (TP5List[0]/Label5Num, TP5List[1]/Label5Num, TP5List[2]/Label5Num))
    print("F1: %f, %f, %f\n" %\
              ((2*TP5List[0]/float(mTest+TP5List[0]-TN5List[0])), (2*TP5List[1]/float(mTest+TP5List[1]-TN5List[1]))\
               , (2*TP5List[2]/float(mTest+TP5List[2]-TN5List[2]))))
    return 
