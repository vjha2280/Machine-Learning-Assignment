from __future__ import division
import scipy.optimize as op
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def CostFunc(theta,X,y):
    lamda = 1
    m,n = X.shape
    Sigmoid = 1/(1+np.exp(-(X.dot(theta.T))))
    L1 = np.log(Sigmoid)
    L2 = np.log(1-Sigmoid)
    J2 = (1/m)*np.sum(-y.T.dot(L1) - ((1-y).T.dot(L2))) +(lamda/(2*m))*np.sum(np.power(theta,2))
    grad = (Sigmoid-y).dot(X)*(1/m) + (lamda/m)*theta
    return J2,grad

if __name__ == "__main__":
    
    data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]
    
    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
        
    theta = np.zeros(n + 1)
             
    theta1, nfeval, rc = op.fmin_tnc(func = CostFunc, x0 = theta, args =(X,y),messages=0)
    
    print(theta1)

    
