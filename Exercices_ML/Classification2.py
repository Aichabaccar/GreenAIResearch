# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:22:28 2022

@author: chouc
"""
import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd

# initialisation du réseau de neurones
def init(I, N, J, data, K):
    
    X=np.zeros([I,N])
    #définition de X
    for i in range(I):
        X[i, 0] = data[i, 0]
        X[i,1] = data[i,1]
    
    #définition de Y
    Y=np.zeros([I,J])
    Ybis=np.zeros(I)
    for i in range(I):
        Ybis[i] = data[i,2]
    for i in range(len(Ybis)):
        col = Ybis[i]
        col = int(col)
        Y[i,col]= 1
        
    #plot les données en les catégorisant
    df= pd.DataFrame(data = data)
    plt.figure(1)
    plt.scatter(df[0], df[1], c=df[2])
    plt.title('Training datas')
    plt.show()
    
    #définition de Xbar
    Xbar=np.zeros([I,(N+1)])
    for i in range(I):
        Xbar[i,0]=1 #x0 = 1
        Xbar[i,1]=X[i,0]
        Xbar[i,2]=X[i,1]
    
    #définition de V
    V=np.random.rand((N+1),K) *0.01
    
    #définition de Xbb
    Xbb = np.zeros([I,K])
    Xbb=Xbar.dot(V)
    
    #définition de F
    F=np.zeros([I,K])
    for i in range(I):
        for j in range(K):
            F[i,j]=1/(1+math.exp(-Xbb[i,j]))
    
    #définition de Fbar
    Fbar = np.zeros([I,(K+1)])
    for i in range(I):
        for j in range(1,K):
            Fbar[i,j]=F[i,j]
            Fbar[i,0]=1
    
    #définition de W   
    W = np.random.rand((K+1),J) * 0.01
    
    #définition de Fbb
    Fbb = np.zeros([I,J])
    Fbb = Fbar.dot(W)
    
    #définition de G
    G =np.zeros([I,J])
    for i in range(I):
        for j in range(J):
            G[i,j]=1/(1+math.exp(-Fbb[i,j]))

    return X, Y, Xbar, Xbb, V, F, W, Fbar, Fbb, G

#front-propagation
def front_prop(I, K, J, Xbb, Xbar, V, W, F, Fbar, Fbb, G):
    #redéfinition de Xbb
    Xbb=Xbar.dot(V)
    
    #redéfinition de F
    for i in range(I):
        for j in range(K):
            F[i,j]=1/(1+math.exp(-Xbb[i,j]))
    
    #redéfinition de Fbar
    for i in range(I):
        for j in range(1,K):
            Fbar[i,j]=F[i,j]
            Fbar[i,0]=1
            
    #redéfinition de Fbb
    Fbb = Fbar.dot(W)
    
    #redéfinition de G
    for i in range(I):
        for j in range(J):
            G[i,j]=1/(1+math.exp(-Fbb[i,j]))
    
    return F, Fbar, Fbb, G, Xbb
    
    
#back-propagation
def back_prop(alpha, alpha2, I, N, J, K, G, V, Y, Fbar, W, Xbar, F):
    #redéfinition de W
    for k in range(K+1):
        grad = 0.0
        for j in range(J):
            for i in range(I):
                grad += (G[i,j]-Y[i,j]) * G[i,j] * (1-G[i,j]) * Fbar[i,k]
            W[k,j] -= alpha * grad
    
    #redéfinition de V
    for n in range(N+1):
        for k in range(K):
            grad2 = 0.0
            for i in range(I):
                for j in range(J):
                    grad2 += ( (G[i,j]-Y[i,j]) * G[i,j] * (1-G[i,j]) * W[k,j] * F[i,k] * (1-F[i,k]) * Xbar[i,n])
            V[n,k] -= alpha2 * grad2
            
    return W, V

#calcul du SSE
def sse(Y, G, I, J):
    somme = 0
    for i in range(I):
        for j in range(J):
            somme += np.abs(G[i,j]-Y[i,j])
    
    return 0.5 * np.power(somme, 2)

#predictions sur des données autres 
def predict(X, V, W,I, N, J, K):
    #définition de Xbar
    Xbar=np.zeros([I,(N+1)])
    for i in range(I):
        Xbar[i,0]=1 #x0 = 1
        Xbar[i,1]=X[i,0]
        Xbar[i,2]=X[i,1]

    
    #définition de Xbb
    Xbb = np.zeros([I,K])
    Xbb=Xbar.dot(V)
    
    #définition de F
    F=np.zeros([I,K])
    for i in range(I):
        for j in range(K):
            F[i,j]=1/(1+math.exp(-Xbb[i,j]))
    
    #définition de Fbar
    Fbar = np.zeros([I,(K+1)])
    for i in range(I):
        for j in range(1,K):
            Fbar[i,j]=F[i,j]
            Fbar[i,0]=1

    
    #définition de Fbb
    Fbb = np.zeros([I,J])
    Fbb = Fbar.dot(W)
    
    #définition de G
    G =np.zeros([I,J])
    
    for i in range(I):
        for j in range(J):
            G[i,j]=1/(1+math.exp(-Fbb[i,j]))
    
    Gtest=np.zeros(I)
    col = np.argmax(G, axis=1)
    for i in range(I):
        Gtest[i]=col[i]
    
    return Gtest     

data = np.loadtxt('data_ffnn_3classes.txt', delimiter = ' ',  dtype=float)
I = 71
N = 2
K = 5
J =3
[X, Y, Xbar, Xbb, V, F, W, Fbar, Fbb,G] = init(I,N,J,data,K)
[W, V] = back_prop(0.1, 0.1, I,N,J,K,G,V,Y,Fbar,W,Xbar,F)
E=np.zeros(5000)
for i in range(5000):
    [F, Fbar, Fbb, G, Xbb]=front_prop(I,K, J, Xbb, Xbar, V, W, F, Fbar, Fbb, G)
    [W, V] = back_prop(0.1, 0.1, I,N,J,K,G,V,Y,Fbar,W,Xbar,F)
    E[i]=sse(Y, G, I, J)
plt.figure(2)
plt.xlabel('SSE')
plt.ylabel('Iterations')
plt.title('Error evolution with iterations')
plt.plot(E)

print("\nBest parameters for W :\n",W)
print("\nBest parameters for V :\n", V)

col = np.argmax(Y, axis=1)
col2=np.argmax(G,axis=1)
for i in range(I):
    
    print("real value : ",col[i],", predicted value : ",col2[i])

xtest=np.zeros([3,2])
xtest[0,0] = 2
xtest[0,1] = 2
xtest[1,0] = 4
xtest[1,1] = 4
xtest[2,0] = 4.5
xtest[2,1] = 1.5
print("test values:\n", xtest)
gtest=predict(xtest, V, W, 3, N, J, K)
datapred= np.zeros([3,N+1])
datapred[:,0]=xtest[:,0]
datapred[:,1]=xtest[:,1]
datapred[:,2]=gtest[:]
df= pd.DataFrame(data = datapred)

plt.figure(3)
plt.scatter(df[0], df[1], c=df[2])
plt.title('Group prédiction')
plt.show()
print ("test values with group prediction:\n",datapred)
