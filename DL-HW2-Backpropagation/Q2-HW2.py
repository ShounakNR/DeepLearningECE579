#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import math


# In[48]:


class NodeL2():
    inputArr=np.zeros((3,1))
    def __init__(self,arr):
        self.inputArr=np.copy(arr)
    def output(self):
        return np.sum(np.square(self.inputArr))
    def localGradient(self):
        return np.multiply(2,self.inputArr)
    def downstream(self,upstream):
        return self.localGradient()*upstream


# In[59]:


class NodeSigmoid():
    inputArr =np.zeros((3,1))
    def __init__(self,arr):
        self.inputArr=np.copy(arr)
    def sigmoid(self):
        g=1/(1+np.exp(-self.inputArr))
        return g
    def localGradient(self):
        return np.multiply((1-self.sigmoid()),self.sigmoid())
    def downstream(self,upstream):
        return np.multiply(self.localGradient(),upstream)


# In[93]:


class NodeMultiply():
    inputArr1=np.zeros((3,3))
    inputArr2=np.zeros((3,1))
    def __init__(self,arr1,arr2):
        self.inputArr1=np.copy(arr1)
        self.inputArr2=np.copy(arr2)
    def output(self):
        return np.dot(self.inputArr1,self.inputArr2)
    def localGradient1(self):
        return np.transpose(self.inputArr2)
    def localGradient2(self):
        return np.transpose(self.inputArr1)
    def downstream1(self,upstream):
        return np.dot(upstream,self.localGradient1())
    def downstream2(self,upstream):
        return np.dot(self.localGradient2(),upstream)


# In[131]:


class Fx():
    W=np.zeros((3,3))
    X=np.zeros((3,1))
    def __init__(self,arr1,arr2):
        self.W = np.copy(arr1)
        self.X = np.copy(arr2)
    def forward (self):
        N1 = NodeMultiply(self.W,self.X)
        N2 = NodeSigmoid(N1.output())
        N3 = NodeL2(N2.sigmoid())
        return N3.output()
    def backward (self):
        N1 = NodeMultiply(self.W,self.X)
        N2 = NodeSigmoid(N1.output())
        N3 = NodeL2(N2.sigmoid())
        dW = N1.downstream1(N2.downstream(N3.downstream(1)))
        dX = N1.downstream2(N2.downstream(N3.downstream(1)))
        return dW,dX


# In[132]:




# In[133]:





if __name__ == "__main__":
    n1 =Fx([[1,0,1],[1,0,1],[0,1,1]],[[1],[0],[0]])
   
    print(n1.backward())



