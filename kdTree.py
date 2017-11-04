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

class KD_node(object):
    def __init__(self, point = None, split = None, \
                 LL = None, RR = None):
        #节点值
        self.point = point
        #节点分割维度
        self.split = split
        #节点左孩子
        self.left = LL
        #节点右孩子
        self.right = RR

def createKDTree(data_list):
    length = len(data_list)
    if length == 0:
        return None
    dimension = len(data_list[0])
    max_var = 0

    split = 0
    for i in range(dimension):
        ll = []
        for t in data_list:
            ll.append(t[i])
        var = computerVariance(ll)
        if var > max_var:
            max_var = var
            split = i
    data_list = sorted(data_list, key = lambda x : x[split])
    point = data_list[int(length / 2)]
    root = KD_node(point, split)
    #递归建立左子树
    root.left = createKDTree(data_list[0:int(length/2)])
    #递归建立右子树
    root.right = createKDTree(data_list[int(length / 2) + 1 : length])
    return root

#计算方差
def computerVariance(arraylist):
    arraylist = array(arraylist);
    for i in range(len(arraylist)):
        arraylist[i] = float(arraylist[i]);
    length = len(arraylist);
    sum1 = arraylist.sum();
    array2 = arraylist * arraylist;
    sum2 = array2.sum();
    mean = sum1 / length;
    variance = sum2 / length - mean ** 2;
    return variance;

#用于计算维度距离 def computerDistance(pt1, pt2):
def computerDistance(pt1, pt2):
    sum = 0.0
    for i in range(len(pt1)):
        sum = sum + (pt1[i] - pt2[i]) ** 2
    return sum ** 0.5

#队列中保存最近k节点
def findNN(root, query, k):
    min_dist = computerDistance(query, root.point)
    node_K = []
    nodeList = []
    temp_root = root

    while temp_root:
        nodeList.append(temp_root)
        dd = computerDistance(query, temp_root.point)
        if len(node_K) < k:
            node_K.append(dd)
        else:
            max_dist = max(node_K)
            if dd < max_dist:
                index = node_K.index(max_dist)
                del(node_K[index])
                node_K.append(dd)
        ss = temp_root.split
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right

    while nodeList:
        back_point = nodeList.pop()
        ss = back_point.split
        max_dist = max(node_K)
        if len(node_K) < k or \
           abs(query[ss] - back_point.point[ss]) < max_dist :
            if query[ss] <= back_point.point[ss]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
            if temp_root:
                nodeList.append(temp_root)
                curDist = computerDistance(temp_root.point,query)
                if max_dist > curDist and len(node_K) == k:
                    index = node_K.index(max_dist)
                    del(node_K[index])
                    node_K.append(curDist)
                elif len(node_K) < k:
                    node_K.append(curDist)
    return node_K

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
    root = createKDTree(trainingMat)
    node_K = findNN(root, tile(0, (1024,1)), 10)
    for i in range(10):
        print node_K[i]
    return 
