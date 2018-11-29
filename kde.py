# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:01:32 2018

@author: MAXNU
"""
import math

import numpy as np

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MU = np.zeros(2)
COV = np.eye(2)

def getData(size):
    data = np.random.multivariate_normal(MU, COV, size)
    return data

def gaussian(x, mu, cov):
    gauss = multivariate_normal(mean=mu, cov=cov)
    return gauss.pdf(x)

def area(data):
    size = 40
    X = []
    for i in range(data.shape[1]):
        x=np.linspace(-4, 4, size)
        X.append(x)
    return np.array(X)

def kernel(x, s, h):
    u = (x - s)/h
    v = h * h
    return gaussian(u,np.zeros(2),np.eye(2))/v

def kde(data, h):
    X = area(data)
    size = [len(X[0]), len(X[1])]
    kdepdf = np.zeros(size)
    n = data.shape[0]
    for i in range(size[0]):
        for j in range(size[1]):
            sum1 = 0
            for k in range(n):
                sum1 = sum1 + kernel([X[0][i],X[1][j]], data[k], h)
            kdepdf[i,j] = sum1/n
    return X, kdepdf
   
# n_set = [1, 16, 256, 10000]
n_set = [1, 16]
h1_set = [0.25, 1.0, 4.0]
fig = plt.figure()

pos = 1
for n in n_set :
    for h1 in h1_set :
        s = "n = %d, h1 = %f" % (n, h1)
        print(s)
        data = getData(n)
        hn = 1/math.sqrt(n)
        X,P = kde(data, hn)
        ax = fig.add_subplot(len(n_set), len(h1_set), pos, projection='3d')
        pos = pos + 1
        px, py = np.meshgrid(X[0], X[1])
        ax.plot_surface(px, py, P, cmap='rainbow')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_title(s)
        
plt.savefig("kde_result.png")
plt.show()