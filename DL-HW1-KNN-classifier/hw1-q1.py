import numpy as np
X= np.array([[0,1,0],[0,1,1],[1,2,1],[1,2,0],[1,2,2],[2,2,2],[1,2,-1],[2,2,3],[-1,-1,-1],[0,-1,-2],[0,-1,1],[-1,-2,1]])
# make X as an array of 12x3 dimension with the training data. first 4 entries are A, next 4 are of B and the last 4 are of C.
Y = np.array([[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3]])
# make Y as the label array where label 1 is A, 2 is B and 3 is C.
Xtest = np.array([[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]])
# Xtest is an array I made that has 12 similar elements of the test point. 
# I am using this to calculate the distance between the point and the training data
print(X[0]-Xtest[0])
dist=np.zeros((12,1))
# initializing the array containing the distances with all 0's in th beginning
for i in range(len(X)):
    d= np.linalg.norm(X[i]-Xtest[i])
    # calculates the euclidean distance (L2) and stores it in the variable d which is then entered into the dist array. 
    dist[i]=d
print(dist)
print(np.argmin(dist))
# np.argmin calculates the minimum element in the array and return the index. 
# We can make the comparison by reading the smallest distances and their indexes and comparing the label array with it
