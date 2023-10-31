# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:28:14 2023

@author: 林漢哲
"""
import numpy as np
from qpsolvers import solve_qp

data = np.loadtxt('iris.txt')
X_train = np.concatenate((data[50:75, 2:4], data[100:125, 2:4]), axis = 0)
X_test =  np.concatenate((data[75:100, 2:4], data[125:, 2:4]), axis = 0)
Y_train = np.ones(50)
Y_train[25:] = -1
Y_test = Y_train.copy()

def Linear_kernel(Xi, Xj):
    K = np.dot(Xi.T, Xj)
    return K

def RBF_kernel(Xi, Xj, sigma):
    norm = (np.linalg.norm(Xi - Xj)) ** 2
    K = np.exp(-norm / (2 * (sigma**2)))
    return K

def Poly_kernel(Xi, Xj, times):
    dot = np.dot(Xi.T, Xj)
    K = dot ** times
    return K


def Cal_Alpha(X_train, Y_train, kernel, C, times, sigma):
    train_num = X_train.shape[0]
    q = -np.ones(Y_train.shape[0])
    A = Y_train.copy()
    b = np.array([0.0])
    G = None
    h = None
    lb = np.zeros(Y_train.shape[0])
    ub = C * np.ones(Y_train.shape[0])
    
    if kernel == 'Linear' :
        train_num = X_train.shape[0]
        temp = X_train.copy()
        temp = np.ones((train_num, train_num))
        for i in range(train_num):
            for j in range(train_num):
                temp[i,j] = Linear_kernel(X_train[i], X_train[j])

        P = np.dot(Y_train.reshape((train_num, 1)), Y_train.reshape((1, train_num))) * temp

    if kernel == 'RBF':
        train_num = X_train.shape[0]
        temp1 = np.ones((train_num, train_num))
        for i in range(train_num):
            for j in range(train_num):
                temp1[i,j] = RBF_kernel(X_train[i], X_train[j], sigma)

        P = np.dot(Y_train.reshape((train_num,1)), Y_train.reshape((1,train_num))) * temp1

    if kernel == 'Poly':
        train_num = X_train.shape[0]
        temp2 = np.ones((train_num, train_num))
        for i in range(train_num):
            for j in range(train_num):
                temp2[i,j] = Poly_kernel(X_train[i], X_train[j], times)
    
        P = np.dot(Y_train.reshape((train_num, 1)), Y_train.reshape((1, train_num))) * temp2      

    alpha = solve_qp(P, q, G, h, A, b, lb, ub, solver = "clarabel")

    eps = 2.2204e-16
    for i in range(alpha.size):
        if alpha[i] >= C - np.sqrt(eps):
            alpha[i] = C
            alpha[i] = np.round(alpha[i], 6)
        elif alpha[i] <= 0 + np.sqrt(eps):
            alpha[i] = 0
            alpha[i] = np.round(alpha[i], 6)
        else:
            alpha[i] = np.round(alpha[i], 6)
        
        #print(f"support vector: alpha = {alpha[i]}")
        
    alpha_sum = np.round(np.sum(alpha), 4)

    return alpha, alpha_sum

def KT_condition(alpha, X_train, Y_train, kernel, times, sigma):
    b_star = 0
    count = 0
    for k in range(alpha.size):
        if alpha[k] > 0:
            temp = 0
            for i in range(X_train.shape[0]):
                if alpha[i] > 0:
                    if kernel == 'Linear':
                        temp += alpha[i] * Y_train[i] * Linear_kernel(X_train[i], X_train[k])

                    if kernel == 'RBF':
                        temp += alpha[i] * Y_train[i] * RBF_kernel(X_train[i], X_train[k], sigma)

                    if kernel == 'Poly':
                        temp += alpha[i] * Y_train[i] * Poly_kernel(X_train[i], X_train[k], times)      

            b_star += (1/Y_train[k]) - temp
            count += 1
    
    b_star = (b_star / count)

    return b_star

                 
def SVM(X_train, Y_train, kernel, C, times, sigma, X_test, Y_test):
    alpha, alpha_sum = Cal_Alpha(X_train, Y_train, kernel, C, times, sigma)
    b_star = KT_condition(alpha, X_train, Y_train, kernel, times, sigma)
    #Y_predict = []
    count = 0 
    for i in range(X_test.shape[0]):
        temp = 0
        for j in range(len(alpha)):
            if alpha[j] > 0:
                if kernel == 'Linear':
                    temp += (alpha[j] * Y_train[j] * Linear_kernel(X_test[i], X_train[j]))

                if kernel == 'RBF':
                    temp += (alpha[j] * Y_train[j] * RBF_kernel(X_test[i], X_train[j], sigma))

                if kernel == 'Poly':
                    temp += (alpha[j] * Y_train[j] * Poly_kernel(X_test[i], X_train[j], times))
        
        D = temp + b_star
        if D >= 0:
            #Y_predict.append(1)
            if Y_test[i] == 1:
                count += 1
        else:
            #Y_predict.append(-1)
            if Y_test[i] == -1:
                count += 1
        
    CR = round((count/len(Y_test)) * 100, 2)
    b_star = np.round(b_star , 4)
    
    return alpha, alpha_sum, b_star, CR


#%%  Linear SVM
# Kernel function: K(xi,xj) = np.dot(xi.T, xj)

Linear_C = np.array([1, 10, 100])
print("Linear SVM: ")
for i in range(len(Linear_C)):
    alpha, alpha_sum, b_star, CR = SVM(X_train, Y_train, 'Linear', Linear_C[i], 1, None, X_test, Y_test)
    print(f"Linear SVM (C = {Linear_C[i]}) total alpha value = {alpha_sum}")
    print(f"First five alpha value = {alpha[0:5]}")
    print(f"b* = {b_star}")
    print(f"CR (C = {Linear_C[i]}) : {CR} %\n")
    
# Workspace allocation error：When P matrix has negative eigenvalue


#%%  RBF kernel-based SVM
# Kernel function: K(xi,xj) = exp((xi - xj)/2*sigma^2)

RBF_sigma = np.array([5, 1, 0.5, 0.1, 0.05])
print("\nRBF kernel-based SVM: ")
for i in range(len(RBF_sigma)):
    alpha, alpha_sum, b_star, CR = SVM(X_train, Y_train, 'RBF', 10.0, None, RBF_sigma[i], X_test, Y_test)
    print(f"RBF SVM (sigma = {RBF_sigma[i]}) total alpha value = {alpha_sum}")
    print(f"First five alpha value ={alpha[0:5]}")
    print(f"b* = {b_star}")
    print(f"CR (sigma = {RBF_sigma[i]}) : {CR} %\n")


#%%  Polynomial kernel-based SVM
# Kernel function: K(xi,xj) = (np.dot(xi.T, xj))^p

Poly_p = np.array([1, 2, 3, 4, 5])
print("\nPolynomial kernel-based SVM: ")
for i in range(len(Poly_p)):
    alpha, alpha_sum, b_star, CR = SVM(X_train, Y_train, 'Poly', 10.0, Poly_p[i], None, X_test, Y_test)
    print(f"Polynomial SVM (p = {Poly_p[i]}) total alpha value = {alpha_sum}")
    print(f"First five alpha value = {alpha[0:5]}")
    print(f"b* = {b_star}")
    print(f"CR (p = {Poly_p[i]}) : {CR} %\n")
