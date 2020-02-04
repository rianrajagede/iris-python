from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""
import pandas as pd

#load
datatrain = pd.read_csv('../Datasets/iris/iris_train.csv')

#change string value to numeric
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.values

#split x and y (feature and target)
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1234)

#hyperparameters
hl = 3
lr = 0.01
num_epoch = 50

#build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

#train
for epoch in range(num_epoch):
    X = torch.Tensor(xtrain).float()
    Y = torch.Tensor(ytrain).long()

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    acc = 100 * torch.sum(Y==torch.max(out.data, 1)[1]).double() / len(Y)
    print(type(acc.item()))
    print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                   %(epoch+1, num_epoch, loss.item(), acc.item()))


"""
SECTION 3 : Testing model
"""
#load
datatest = pd.read_csv('../Datasets/iris/iris_test.csv')

#change string value to numeric
datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatest_array = datatest.values

#split x and y (feature and target)
xtest = datatest_array[:,:4]
ytest = datatest_array[:,4]

#get prediction
X = torch.Tensor(xtest).float()
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of the network %.4f %%' % (100 * torch.sum(Y==predicted).double() / len(Y)))