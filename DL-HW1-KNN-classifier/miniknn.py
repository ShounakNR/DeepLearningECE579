import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')
# print(mini_train_label)
# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################
    dist=np.zeros((10,40))
    for i in range(len(newInput)):
        for j in range(len(dataSet)):
            d= np.linalg.norm(dataSet[j]-newInput[i])
            # as done before, we are making a 10x40 matrix of the distances between the test points and the training data. 
            # Every row corresponds to 1 test data point and the 40 entries inside it are its distances from the 40 training points. 
            dist[i,j] = d
    
   

    # below are making an array called votes. THis array will have as many elements as there are classification classes. 
    # In this case the classses are 4 in number 

    for i in range(len(newInput)):
        votes = np.zeros(4)
        x= np.argsort(dist[i])[:k]
        # selects the k smallest values and returns an array with the indices of these values in the original unsorted array
        # print(x)
        for i in range(len(x)):
            num_label = labels[x[i]]
            votes[num_label]+=1      
        # print(votes)
        result.append(np.argmax(votes))
        # selects the index with the max value and returns it and appends it to the result array.
    ####################
    # End of your code #
    ####################
    return result

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,6)

print(outputlabels)
print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")
