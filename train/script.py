### Import Libraries
import numpy as np
import torch
import torchvision
from time import time

from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F



### Load Data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('./dataset/trainset', download=True, train=True, transform=transform)
valset = datasets.MNIST('./dataset/valset', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
print("Data loaded")



### Build Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
model = Net()



### Train Model
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

n_epochs = 10
time0 = time()

for epoch in range(n_epochs):
    train_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()
    else:
        train_loss = train_loss/len(trainloader.dataset)
        print('Epoch: {} \tTraining loss: {:.6f}'.format(
            epoch+1,
            train_loss
        ))
        
print("\nTraining Time (in minutes) =",(time()-time0)/60)



### Validate Model
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    with torch.no_grad():
        logps = model(images)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))



### Save Model
print("Save model")
torch.save(model, "saved_model.pth")