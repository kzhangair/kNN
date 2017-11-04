# -*- coding: utf-8 -*-

from numpy import *
import operator
from os import listdir

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

#马氏距离（Mahalanobis Distance）计算函数
def MaDistances(inX, dataSet, invCovMat):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    diffMat.shape = (dataSetSize, 1024)
    sqDistances = []
    for i in range(dataSetSize):
        sqDistances.append(diffMat[i].dot(invCovMat).dot(diffMat[i].T))
    sqDistances = array(sqDistances)
    distances = sqDistances ** 0.5
    return distances
    
    
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

#对输入inX，基于训练样本集(dataSet, labels)用k近邻做出判决
#距离度量计算函数由参数DistanceCalFunc给出
#def DistanceCalFunc(inX, dataSet):
#   ...
#   return distances(dataSetSize*1的ndarray) 
#返回：inX的判决结果
def classify0(inX, dataSet, labels, k, DistanceCalfunc, covInvMat=None):
    if(DistanceCalfunc.__name__ == "MaDistances"):
        distances = DistanceCalfunc(inX, dataSet,covInvMat)
    elif(DistanceCalfunc.__name__ == "EuDistances"):
        distances = DistanceCalfunc(inX, dataSet)    
    sortedDistIndicies = distances.argsort() #return index
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        kWeightedDist = WeightCal2(sortedDistIndicies, distances, i, k)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + kWeightedDist #default:0
    sortedClassCount = sorted(classCount.iteritems(), \
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

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
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, \
                                     trainingMat, hwLabels, 3, EuDistances)
        print "the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, classNumStr)
        if(classifierResult != classNumStr):
            errorCount += 1.0
        print "\nthe total number of errors is: %d" % errorCount
        print "\nthe total error rate is: %f" % (errorCount/float(mTest))
