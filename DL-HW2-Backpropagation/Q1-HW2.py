

import math

import numpy as np


class NodeInverse ():
    input = 1
    def __init__(self,num):
        self.input=num
    def output(self):
        result =1/self.input
        return result
    def localGradient(self):
        return -1*math.pow(self.input,-2)
    def downstream(self,upstream):
        return self.localGradient()*upstream



class NodeLinear ():
    input = 1
    b=0
    def __init__(self,num,b):
        self.input=num
        self.b = b
    def output(self):
        result = self.input+self.b
        return result
    def localGradient(self):
        return 1
    def downstream(self,upstream):
        return self.localGradient()*upstream



class NodeAdd ():
    input1 = 0
    input2=0
    def __init__(self,num1,num2):
        self.input1=num1
        self.input2=num2
    def output(self):
        result = self.input1+self.input2
        return result
    def localGradient(self):
        return 1
    def downstream(self,upstream):
        return self.localGradient()*upstream


class NodeSine ():
    input = 1
    def __init__(self,num):
        self.input=num
    def output(self):
        result = math.sin(self.input)
        return result
    def localGradient(self):
        return math.cos(self.input)
    def downstream(self,upstream):
        return self.localGradient()*upstream



class NodeCosine ():
    input = 1
    def __init__(self,num):
        self.input=num
    def output(self):
        result = math.cos(self.input)
        return result
    def localGradient(self):
        return -math.sin(self.input)
    def downstream(self,upstream):
        return self.localGradient()*upstream

class NodeMultiply ():
    input1 = 1
    input2 =1
    def __init__(self,num1,num2):
        self.input1=num1
        self.input2=num2
    def output(self):
        result = self.input1*self.input2
        return result
    def localGradient1(self):
        return self.input2
    def localGradient2(self):
        return self.input1
    def downstream1(self,upstream):
        return self.localGradient1()*upstream
    def downstream2(self,upstream):
        return self.localGradient2()*upstream


class Fx():
    W1=0
    W2=0
    X1=0
    X2=0
    def __init__(self,a,b,c,d):
        self.W1=a
        self.X1=b
        self.W2=c
        self.X2=d
    def forward (self):
        N1=NodeMultiply(self.W1,self.X1)
        N2=NodeSine(N1.output())
        N7=NodeMultiply(self.W1,self.X1)
        N8=NodeSine(N7.output())
        N3=NodeMultiply(N2.output(),N8.output())
        N9 = NodeMultiply(self.W2,self.X2)
        N10=NodeCosine(N9.output())
        N4 = NodeAdd(N3.output(),N10.output())
        N5 = NodeLinear(N4.output(),2)
        N6 = NodeInverse(N5.output())
        result= N6.output()
        return result
    def backward (self):
        N1=NodeMultiply(self.W1,self.X1)
        N2=NodeSine(N1.output())
        N7=NodeMultiply(self.W1,self.X1)
        N8=NodeSine(N7.output())
        N3=NodeMultiply(N2.output(),N8.output())
        N9 = NodeMultiply(self.W2,self.X2)
        N10=NodeCosine(N9.output())
        N4 = NodeAdd(N3.output(),N10.output())
        N5 = NodeLinear(N4.output(),2)
        N6 = NodeInverse(N5.output())
        dW1= N1.downstream1(N2.downstream(N3.downstream1(N4.downstream(N5.downstream(N6.downstream(1))))))
        dX1= N1.downstream2(N2.downstream(N3.downstream1(N4.downstream(N5.downstream(N6.downstream(1))))))
        dW2= N9.downstream1(N10.downstream(N4.downstream(N5.downstream(N6.downstream(1)))))
        dX2= N9.downstream2(N10.downstream(N4.downstream(N5.downstream(N6.downstream(1)))))
#         N1.downstream1(N2.downstream(N3.downstream1(N4.downstream(N5.downstream(N6.downstream(1))))))
        return [dW1,dX1,dW2,dX2]
    

if __name__ == "__main__":
    a=Fx(2,-1,-3,-2)
    print(a.backward())





