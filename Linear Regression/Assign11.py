# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:53:22 2018

@author: vinita
"""

import numpy as np
import pandas as pd

if __name__ == "__main__":

    data = np.loadtxt(open("C:\Coursera-Assignment-master\Machine-Learning-Assignment\Linear Regression\ex1data1.txt", "r"),delimiter=",")
    X = data[:,0:1]
    y = data[:,1]
    m , n = X.shape
    theta = np.zeros(n+1)
    X = np.hstack((np.ones((m, 1)), X))
    alpha = 0.01
    J1 = 0
    converge = False
    while not converge:
        J2 = np.sum(np.power(X.dot(theta.T)- y,2))/ (2 * m)
        if abs(J1 - J2) > 0.000001:
            grad = (alpha / m) * (X.dot(theta.T)-y).T.dot(X)
            theta = theta - grad
            J1 = J2
        else:
            converge =True

print (theta)
print (J2)