import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
# x_train = np.load('../x_train.npy')
# y_train = np.load('../y_train.npy')
# x_test = np.load('../x_test.npy')
# y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
# print(y_test[0:10])
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    test_len = len(newInput)
    train_len= len(dataSet)

    dist=np.zeros((test_len,train_len))
    for i in range(test_len):
        for j in range(train_len):
            d= np.linalg.norm(dataSet[j]-newInput[i])
            dist[i,j] = d
    
    # print(dist)
    
    # print(labels)
    for i in range(test_len):
        votes = np.zeros(10)
        x= np.argsort(dist[i])[:k]
        # print(x)
        for i in range(len(x)):
            num_label = labels[x[i]]
            # print(num_label)
            votes[num_label]+=1      
        # print(votes)
        result.append(np.argmax(votes))
     
    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,12)
print(outputlabels)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
