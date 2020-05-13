import numpy as np
import torch
import torchvision

from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])     


trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)      


trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)                 


testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)      


testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)                   



model=nn.Sequential(nn.Linear(784,200), # 1 layer:- 784 input 128 output
                    nn.ReLU(),          # Defining Regular linear unit as activation function
                    nn.Linear(200,50),  # 2 Layer:- 128 Input and 64 output
                    nn.ReLU(),          # Defining Regular linear unit as activation function
                    nn.Linear(50,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)
                    # nn.Softmax() # Defining the log softmax to find the probablities for the last output unit
                  ) 

# Loss function: Cross Entropy loss as it implements Softmax classifier followed by NLL internally
loss_function = nn.CrossEntropyLoss() 

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


start_time = time()
epochs = 10 

for epoch in range(epochs):
    total_loss = 0
    for images, labels in trainloader:

        images = images.view(images.shape[0], -1) 

        optimizer.zero_grad()

        output = model(images)

        loss = loss_function(output, labels)
        
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        
    else:
        print("Epoch {} - Training loss: {}".format(epoch, total_loss/len(trainloader)))
    
print("\nTraining Time (in minutes) =",(time()-start_time)/60)


correct_count, all_count = 0, 0

for images,labels in testloader:
    for i in range(len(images)):
      img = images[i].view(1, 784)
      with torch.no_grad():
        model_output = model(img)

      probability_dist = torch.exp(model_output)
      probability_array = np.array(probability_dist.numpy()[0])
      probability_normalized = probability_array/np.sum(probability_array)

      pred_label = np.argmax(probability_normalized)

      true_label = labels.numpy()[i]
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count)*100)