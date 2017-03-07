"""
SECTION 1 : Load and setup data for training
"""

import csv
import random
import math
random.seed(123)

# Load dataset
with open('../Datasets/iris/iris.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None) # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in xrange(len(row))]

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

optimizer = stochastic gradient descent
loss function = Square Root Error
learning rate = 0.007
epoch = 300

best result = 100%
"""

def matrix_mul_bias(A, B, bias): # matrix multiplication +  bias
    C = [[0 for i in xrange(len(B[0]))] for i in xrange(len(A))]    
    for i in xrange(len(A)):
        for j in xrange(len(B[0])):
            for k in xrange(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def mat_vec(A, B): # matrix x vector (for backprop)
    C = [0 for i in xrange(len(A))]
    for i in xrange(len(A)):
        for j in xrange(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A): # activation function: sigmoid
    for i in xrange(len(A)):
        for j in xrange(len(A[0])):
            A[i][j] = 1 / (1 + math.exp(-A[i][j]))
    return A

# Define parameter
alfa = 0.007
epoch = 300
neuron = [4, 3] # architecture, number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in xrange(neuron[1])] for i in xrange(neuron[0])]
bias = [0 for i in xrange(neuron[1])]

# Initiate weight with random between -1.0 ... 1.0
for i in xrange(neuron[0]):
    for j in xrange(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for e in xrange(epoch):
    cost_total = 0

    # Forward propagation
    h_1 = matrix_mul_bias(train_X, weight, bias)
    X_1 = sigmoid(h_1)

    for idx, x in enumerate(train_X): # Update for each data; SGD      
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in xrange(3):
            eror +=  0.5 * (target[i] - X_1[idx][i]) ** 2 
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta = []
        for j in xrange(neuron[1]):
            delta.append(-1 * (target[j]-X_1[idx][j]) * X_1[idx][j] * (1-X_1[idx][j]))

        for i in xrange(neuron[0]):
            for j in xrange(neuron[1]):
                weight[i][j] -= alfa * (delta[j] * x[i])
                bias[j] -= alfa * delta[j]
        
        # # Update weight and bias (layer 1)
        # delta_1 = mat_vec(weight_2, delta)
        # for j in xrange(neuron[1]):
        #     delta_1[j] = delta_1[j] * (X_1[idx][j] * (1-X_1[idx][j]))
        
        # for i in xrange(neuron[0]):
        #     for j in xrange(neuron[1]):
        #         weight[i][j] -=  alfa * (delta_1[j] * x[i])
        #         bias[j] -= alfa * delta_1[j]
    
    cost_total /= len(train_X)
    if(e % 80 == 0):
        print cost_total 

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)

# Get prediction
print [int(y) for y in test_y]
preds = []
for r in res:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print preds

# Calculate accuration
acc = 0.0
for i in xrange(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print acc / len(preds) * 100, "%"