import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

dataiter = iter(trainloader) # creating a iterator
images, labels = dataiter.next() # creating images for image and lables for image number (0 to 9) 


model=nn.Sequential(nn.Linear(784,200), # 1 layer:- 784 input 128 o/p
                    nn.ReLU(),          # Defining Regular linear unit as activation
                    nn.Linear(200,50),  # 2 Layer:- 128 Input and 64 O/p
                    nn.ReLU(),          # Defining Regular linear unit as activation
                    nn.Linear(50,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)
                    # nn.Softmax() # Defining the log softmax to find the probablities for the last output unit
                  ) 



criterion = nn.CrossEntropyLoss()

# images, labels = next(iter(trainloader))
# images = images.view(images.shape[0], -1)

# logps = model(images) #log probabilities
# loss = criterion(logps, labels) #calculate the NLL-loss

# print('Before backward pass: \n', model[0].weight.grad)
# loss.backward() # to calculate gradients of parameter 
# print('After backward pass: \n', model[0].weight.grad)


# defining the optimiser with stochastic gradient descent and default parameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# print('Initial weights - ', model[0].weight)

# images, labels = next(iter(trainloader))
# images.resize_(128, 784)

# # Clear the gradients, do this because gradients are accumulated
# optimizer.zero_grad()

# # Forward pass
# output = model(images)
# loss = criterion(output, labels)
# # the backward pass and update weights
# loss.backward()
# print('Gradient -', model[0].weight.grad)


time0 = time()
epochs = 10 # total number of iteration for training
running_loss_list= []
epochs_list = []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatenning MNIST images with size [128,784]
        images = images.view(images.shape[0], -1) 
    
        # defining gradient in each epoch as 0
        optimizer.zero_grad()
        
        # modeling for each image batch
        output = model(images)
        
        # calculating the loss
        loss = criterion(output, labels)
        
        # This is where the model learns by backpropagating
        loss.backward()
        
        # And optimizes its weights here
        optimizer.step()
        
        # calculating the loss
        running_loss += loss.item()
        
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))