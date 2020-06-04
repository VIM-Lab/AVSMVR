import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

import params


# 用于可视化图像
def plt_image(x):
    plt.imshow(x)
    plt.show()


# 用于可视化展示体素
def plt_voxel(y):
    if y.shape[0] == 1: # batch为1的情况下，去掉batch维
        y = y[0]

    if y.ndim == 4: # onehot编码时，取出真实体素
        y = y[:, :, :, -1]

    y = np.swapaxes(y, 1, 2) # 交换两个维度，使得可视化的角度更加符合需求

    # 可视化三维体素
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(y)
    plt.show()


# 用于保存体素图片
def save_voxel(y, name):
    if y.shape[0] == 1: # batch为1的情况下，去掉batch维
        y = y[0]

    if y.ndim == 4: # onehot编码时，取出真实体素
        y = y[:, :, :, -1]

    y = np.swapaxes(y, 1, 2) # 交换两个维度，使得可视化的角度更加符合需求

    # 可视化三维体素
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(y)
    plt.savefig(os.path.join(params.save_path, name))
    plt.close('all')


def cal_iou(gt, pre): # shape=[32, 32, 32]
    inter = 0
    union = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i,j,k] and pre[i,j,k]:
                    union += 1
                    inter += 1
                elif gt[i,j,k] or pre[i,j,k]:
                    union += 1
    return inter / union


# 用于将体素预测概率转换为0/1
def dicide_voxel(voxel): # input shape = [1, 32, 32, 32, 2]
    voxel = voxel[0]
    voxel = voxel.numpy()
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[0]):
                if voxel[i,j,k,1] >= voxel[i,j,k,0]:
                    voxel[i,j,k,1] = 1
                    voxel[i,j,k,0] = 0
                else:
                    voxel[i,j,k,1] = 0
                    voxel[i,j,k,0] = 1

    return voxel


def predict_occupation(pre): # pre.shape=[1, 32, 32, 32, 2]
    pre = pre[0] # pre.shape=[32, 32, 32, 2]
    occupy = np.array([[[0 for dim1 in range(32)] for dim2 in range(32)] for dim3 in range(32)])
    threshold = np.array([[[0.31 for dim1 in range(32)] for dim2 in range(32)] for dim3 in range(32)])
    for i in range(16):
        for j in range(i, 32-i):
            for k in range(i, 32-i):
                if pre[i,j,k,1] >= threshold[i,j,k]:
                    occupy[i,j,k] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[i+1, j+_j, k+_k] += 0.01
                if pre[31-i,j,k,1] >= threshold[31-i,j,k]:
                    occupy[31-i,j,k] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[30-i, j+_j, k+_k] += 0.01
        for j in range(i+1, 31-i):
            for k in range(i, 32-i):
                if pre[j,i,k,1] >= threshold[j,i,k]:
                    occupy[j,i,k] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[j+_j, i+1, k+_k] += 0.01
                if pre[j,31-i,k,1] >= threshold[j,31-i,k]:
                    occupy[j,31-i,k] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[j+_j, 30-i, k+_k] += 0.01
        for j in range(i+1, 31-i):
            for k in range(i+1, 31-i):
                if pre[j,k,i,1] >= threshold[j,k,i]:
                    occupy[j,k,i] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[j+_j, k+_k, i+1] += 0.01
                if pre[j,k,31-i,1] >= threshold[j,k,31-i]:
                    occupy[j,k,31-i] = 1
                    for _j in range(-1, 2):
                        for _k in range(-1, 2):
                            if j+_j >= i+1 and j+_j <= 30-i and k+_k >= i+1 and k+_k <= 30-i:
                                threshold[j+_j, k+_k, 30-i] += 0.01
    
    return occupy


# 用于排序输入图像
def x_sort(x):
    s = np.array([[_, 0] for _ in range(24)])

    for image in range(24):
        s[image, 1] = np.sum(x[image])

    s = sorted(s , key = lambda x : x[1], reverse=True)
    s = [_[0] for _ in s]
    
    x = x[s]
    return x
    

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def choose_action(prob, used):
    prob = np.sum(prob, 0)

    prob[used] = 0.0 # 已用过的action概率置零
    prob = softmax(prob) # 归一化

    res = np.random.choice(np.arange(24), p=prob.ravel())
    return res

def random_action():
    res = np.random.choice(np.arange(24))
    return res

