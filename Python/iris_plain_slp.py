from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training
"""

import csv
import random
import math
random.seed(113)

# Load dataset
with open('../Datasets/iris/iris.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None) # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
SECTION 2 : Build and Train Model
Single layer perceptron model
input layer : 4 neuron, represents the feature of Iris
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 76.67%
"""

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
        C[j] += bias[j]
    return C


def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False): 
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 3] # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
bias = [0 for i in range(neuron[1])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(neuron[1]):
            eror += (target[i] - X_1[i]) ** 2
        cost_total += eror * 1 / neuron[1]

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta = []
        for j in range(neuron[1]):
            delta.append(2. / neuron[1] * (target[j]-X_1[j]) * -1  * X_1[j] * (1-X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alfa * (delta[j] * x[i])
                bias[j] -= alfa * delta[j]

    cost_total /= len(train_X)
    if(e % 100 == 0):
        print(cost_total)

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)

# Get prediction
preds = []
for r in res:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print(preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print(acc / len(preds) * 100, "%")