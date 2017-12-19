# -*- coding: utf-8 -*-

from numpy import *
import operator
from os import listdir
import time
import matplotlib.pyplot as plt

    
    
#欧式距离（Mahalanobis distance）计算函数    
def EuDistances(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 #each element **2
    sqDistances = sqDiffMat.sum(axis=1) #row sum to one element
    distances = sqDistances ** 0.5
    return distances

#曼哈顿距离（Manhattan distance）
def ManDistances(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    absDiffMat = abs(diffMat) 
    distances = absDiffMat.sum(axis=1)
    return distances

#L距离(p=infinity)
def LInftyDistances(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    absDiffMat = abs(diffMat) 
    distances = absDiffMat.max(1)
    return distances

#马氏距离（Mahalanobis distance）计算函数
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

#对输入inX，基于训练样本集(dataSet, labels)用k近邻做出判决
#距离度量计算函数由参数DistanceCalFunc给出
#返回：k取K=[k1, k2, ..., kn]情况下的inX的判决结果向量resultList=[result1, result2, ..., resultn]
def classify0(inX, dataSet, labels, K, DistanceCalfunc, covInvMat=None):
    if(DistanceCalfunc == "MaDistances"):
        distances = MaDistances(inX, dataSet,covInvMat)
    elif(DistanceCalfunc == "EuDistances"):
        distances = EuDistances(inX, dataSet)
    elif(DistanceCalfunc == "ManDistances"):
        distances = ManDistances(inX, dataSet)
    elif(DistanceCalfunc == "LInftyDistances"):
        distances = LInftyDistances(inX, dataSet)
    sortedDistIndicies = distances.argsort() #return index
    classCount = {}
    resultList = []
    j=0
    for i in range(K[-1]+1):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #default:0
        if(i == 4*j+3):        
            sortedClassCount = sorted(classCount.iteritems(),
                                      key=operator.itemgetter(1), reverse=True)
            resultList.append((int)(sortedClassCount[0][0]))
            j += 1
    return resultList

def handwritingClassTest():
    #使用img2vector转化图像数据为1*1024的向量，存储到trainingMat矩阵（m*1024）中，m为训练数据集中的样本数量
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)               #trainingDigits文件夹中的训练样本数量
    trainingMat = zeros((m, 1024))          #构造一个m*1024的矩阵，用来存储训练样本  
    for i in range(m):
        fileNameStr = trainingFileList[i]   #一个样本的文件名
        fileStr = fileNameStr.split('.')[0] #只取文件名，去掉后缀txt
        classNumStr = int(fileStr.split('_')[0])#取出类别
        hwLabels.append(classNumStr)        #将Label添加到hwLabels
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) #转化图像数据为1*1024向量

    #选取准确率、召回率、F1度量和算法速度作为性能指标
    #精度矩阵
    correctRateMat = zeros((4*10)).reshape(4,10)
    #准确率矩阵
    precisionRateMat = zeros((4*10)).reshape(4,10)
    #召回率矩阵
    recallRateMat = zeros((4*10)).reshape(4,10)
    #F1度量矩阵
    F1Mat = zeros((4*10)).reshape(4,10)
    
    testFileList = listdir('testDigits')
    invCovMat = linalg.pinv(cov(trainingMat.T))             #计算训练集矩阵协方差矩阵的逆，用于计算马氏距离
    mTest = len(testFileList)

    #然后对每个测试样本（testDigits/文件夹），先用img2vector函数转换数据格式，然后用classify0进行判决，统计各项性能指标
    DistanceCalfunc = ["EuDistances", "ManDistances", "LInftyDistances", "MaDistances"]
    K = range(3, 43, 4)
    for j in range(4):
        start = time.clock()
        #正确个数
        correctCntList=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #数字5识别的TP值
        TP5List= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #所有判决为5的结果数量 TP+FP
        P5List = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        #数字5识别的TN值
        TN5List = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #测试样本中数字5的样本数量
        Label5Num = 107.0
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
            if(j <= 2):
                classifierResultList = classify0(vectorUnderTest, \
                                       trainingMat, hwLabels, K, DistanceCalfunc=DistanceCalfunc[j])
            elif (j == 3):
                classifierResultList = classify0(vectorUnderTest, \
                                       trainingMat, hwLabels, K, DistanceCalfunc[j], invCovMat)
            #正确分类的样本数，TP, TN, TP+FP
            for s in range(10):
                if(classifierResultList[s] == classNumStr):
                    correctCntList[s] += 1.0
                if(classifierResultList[s] == 5):
                    P5List[s] += 1
                    if(classNumStr == 5):
                        TP5List[s] += 1
                else:
                    if(classNumStr != 5):
                        TN5List[s] += 1
        elapsed = (time.clock() - start)
        print("%s Time used: %d秒" % (DistanceCalfunc[j], elapsed))
        print("k = 3,7,11,15,19,23,27,31,35,39")
        print("correct rate: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f" %\
              (correctCntList[0]/float(mTest),correctCntList[1]/float(mTest), correctCntList[2]/float(mTest)\
            , correctCntList[3]/float(mTest),correctCntList[4]/float(mTest), correctCntList[5]/float(mTest) \
            , correctCntList[6]/float(mTest),correctCntList[7]/float(mTest), correctCntList[8]/float(mTest) \
            , correctCntList[9]/float(mTest)))
        print("precision rate:%f, %f, %f, %f, %f, %f, %f, %f, %f, %f" %\
               (TP5List[0]/float(P5List[0]),TP5List[1]/float(P5List[1]),TP5List[2]/float(P5List[2])\
               ,TP5List[3]/float(P5List[3]),TP5List[4]/float(P5List[4]),TP5List[5]/float(P5List[5])\
               ,TP5List[6]/float(P5List[6]),TP5List[7]/float(P5List[7]),TP5List[8]/float(P5List[8]),TP5List[9]/float(P5List[9])))
        print("recall rate: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f" %\
              (TP5List[0]/Label5Num, TP5List[1]/Label5Num, TP5List[2]/Label5Num, TP5List[3]/Label5Num, \
                              TP5List[4]/Label5Num, TP5List[5]/Label5Num, TP5List[6]/Label5Num, TP5List[7]/Label5Num, \
                              TP5List[8]/Label5Num, TP5List[9]/Label5Num))
        print("F1: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" %\
              ((2*TP5List[0]/float(mTest+TP5List[0]-TN5List[0])), (2*TP5List[1]/float(mTest+TP5List[1]-TN5List[1]))\
               , (2*TP5List[2]/float(mTest+TP5List[2]-TN5List[2])), (2*TP5List[3]/float(mTest+TP5List[3]-TN5List[3]))\
               , (2*TP5List[4]/float(mTest+TP5List[4]-TN5List[4])), (2*TP5List[5]/float(mTest+TP5List[5]-TN5List[5]))\
               , (2*TP5List[6]/float(mTest+TP5List[6]-TN5List[6])), (2*TP5List[7]/float(mTest+TP5List[7]-TN5List[7]))\
               , (2*TP5List[8]/float(mTest+TP5List[8]-TN5List[8])), (2*TP5List[9]/float(mTest+TP5List[9]-TN5List[9]))))
        correctRateMat[j,:] = array([[correctCntList[0]/float(mTest), correctCntList[1]/float(mTest), correctCntList[2]/float(mTest)\
            , correctCntList[3]/float(mTest), correctCntList[4]/float(mTest), correctCntList[5]/float(mTest) \
            , correctCntList[6]/float(mTest), correctCntList[7]/float(mTest), correctCntList[8]/float(mTest) \
            , correctCntList[9]/float(mTest)]]).reshape(1, 10)
        precisionRateMat[j,:] = array([TP5List[0]/float(P5List[0]),TP5List[1]/float(P5List[1]),TP5List[2]/float(P5List[2])\
               ,TP5List[3]/float(P5List[3]),TP5List[4]/float(P5List[4]),TP5List[5]/float(P5List[5])\
               ,TP5List[6]/float(P5List[6]),TP5List[7]/float(P5List[7]),TP5List[8]/float(P5List[8]),TP5List[9]/float(P5List[9])]) 
        recallRateMat[j,:] = array([TP5List[0]/Label5Num, TP5List[1]/Label5Num, TP5List[2]/Label5Num, TP5List[3]/Label5Num, \
                              TP5List[4]/Label5Num, TP5List[5]/Label5Num, TP5List[6]/Label5Num, TP5List[7]/Label5Num, \
                              TP5List[8]/Label5Num, TP5List[9]/Label5Num])
        F1Mat[j,:] = array([(2*TP5List[0]/float(mTest+TP5List[0]-TN5List[0])), (2*TP5List[1]/float(mTest+TP5List[1]-TN5List[1]))\
               , (2*TP5List[2]/float(mTest+TP5List[2]-TN5List[2])), (2*TP5List[3]/float(mTest+TP5List[3]-TN5List[3]))\
               , (2*TP5List[4]/float(mTest+TP5List[4]-TN5List[4])), (2*TP5List[5]/float(mTest+TP5List[5]-TN5List[5]))\
               , (2*TP5List[6]/float(mTest+TP5List[6]-TN5List[6])), (2*TP5List[7]/float(mTest+TP5List[7]-TN5List[7]))\
               , (2*TP5List[8]/float(mTest+TP5List[8]-TN5List[8])), (2*TP5List[9]/float(mTest+TP5List[9]-TN5List[9]))])
    #使用Matplotlib画图
    plt.figure(num='Experiment Result')
    plt.subplot(2,2,1)
    plt.title('correct rate')
    plt.xlabel("k")
    plt.ylabel("correct rate")
    plt.axis([0.0, 40.0, 0.0, 1.0])
    plt.plot(K, correctRateMat[0,:], label=DistanceCalfunc[0])
    plt.plot(K, correctRateMat[1,:], label=DistanceCalfunc[1])
    plt.plot(K, correctRateMat[2,:], label=DistanceCalfunc[2])
    plt.plot(K, correctRateMat[3,:], label=DistanceCalfunc[3])
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,2)
    plt.title('precision rate')
    plt.xlabel("k")
    plt.ylabel("precision rate")
    plt.axis([0.0, 40.0, 0.0, 1.0])
    plt.plot(K, precisionRateMat[0,:], label=DistanceCalfunc[0])
    plt.plot(K, precisionRateMat[1,:], label=DistanceCalfunc[1])
    plt.plot(K, precisionRateMat[2,:], label=DistanceCalfunc[2])
    plt.plot(K, precisionRateMat[3,:], label=DistanceCalfunc[3])
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,3)
    plt.title('recall rate')
    plt.xlabel("k")
    plt.ylabel("recall rate")
    plt.axis([0.0, 40.0, 0.0, 1.0])
    plt.plot(K, recallRateMat[0,:], label=DistanceCalfunc[0])
    plt.plot(K, recallRateMat[1,:], label=DistanceCalfunc[1])
    plt.plot(K, recallRateMat[2,:], label=DistanceCalfunc[2])
    plt.plot(K, recallRateMat[3,:], label=DistanceCalfunc[3])
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,4)
    plt.title('F1')
    plt.xlabel("k")
    plt.ylabel("F1")
    plt.axis([0.0, 40.0, 0.0, 1.0])
    plt.plot(K, F1Mat[0,:], label=DistanceCalfunc[0])
    plt.plot(K, F1Mat[1,:], label=DistanceCalfunc[1])
    plt.plot(K, F1Mat[2,:], label=DistanceCalfunc[2])
    plt.plot(K, F1Mat[3,:], label=DistanceCalfunc[3])
    plt.grid(True)
    plt.legend()

    plt.show()
    return 

