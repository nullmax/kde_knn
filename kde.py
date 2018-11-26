# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:01:32 2018

@author: MAXNU
"""
import math

import numpy as np

from scipy import io
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readData():
#    dataFile = ['data/1.mat', 'data/2.mat', 'data/3.mat']
    dataFile = ['data/1.mat']
    x = []
    y = []
    for file in dataFile:
        dataStruct = io.loadmat(file)
        x.extend(dataStruct['x'])
        y.extend(dataStruct['y'])

    x = np.squeeze(x)
    y = np.squeeze(y)
    xlist = np.array([x,y]).transpose()
    return xlist

def gaussian(x, mu, cov):
    gauss = multivariate_normal(mean=mu, cov=cov)
    return gauss.pdf(x)

def area(data):
    size = 30
    X = []
    for i in range(data.shape[1]):
        high = data[:,i].max()
        low = data[:,i].min()
        x=np.linspace(low, high, size)
        X.append(x)
    return np.array(X)

def kernel(x, s, h):
    u = (x - s)/h
    v = h * h
    return gaussian(u,np.zeros(2),np.eye(2))/v

def kde(data, h):
    X = area(data)
    size = [len(X[0]), len(X[1])]
    kpdf = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            sum1 = 0
            for k in range(n):
                sum1 = sum1 + kernel([X[0][i],X[1][j]], data[k], h)
            kpdf[i,j] = sum1/n
    return X, kpdf
    
data = readData()
n = len(data)
hn = 1/math.sqrt(n)

X,P = kde(data, hn)

fig = plt.figure()
ax = Axes3D(fig)
plt.title("KDE")
px, py = np.meshgrid(X[0], X[1])
ax.plot_surface(px, py, P, cmap='rainbow')
plt.show()