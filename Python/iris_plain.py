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
    row[4]=["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4]=[float(row[j]) for j in xrange(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset)*0.8)]
datatest = dataset[int(len(dataset)*0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris

optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.01
epoch = 500
"""

def matrix_mul_bias(A, B, bias): # Fungsi perkalian matrix + bias (untuk Testing)
    C = [[0 for i in xrange(len(B[0]))] for i in xrange(len(A))]    
    for i in xrange(len(A)):
        for j in xrange(len(B[0])):
            for k in xrange(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias=None): # Fungsi perkalian vector dengan matrix + bias
    C = [0 for i in xrange(len(B[0]))]
    for j in xrange(len(B[0])):
        for k in xrange(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B): # Fungsi perkalian matrix dengan vector (untuk backprop)
    C = [0 for i in xrange(len(A))]
    for i in xrange(len(A)):
        for j in xrange(len(B)):
            C[i] += A[i][j] * B[j]

    return C

def sigmoid(A, deriv=False): # Fungsi aktivasi sigmoid
    if deriv: # kalau sedang backprop pakai turunan sigmoid
        for i in xrange(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in xrange(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.01
epoch = 500
neuron = [4, 3, 3] # arsitektur tiap layer

# Initiate weight and bias, random normal antara -1..1
weight = [[0 for j in xrange(neuron[1])] for i in xrange(neuron[0])]
weight_2 = [[0 for j in xrange(neuron[2])] for i in xrange(neuron[1])]
bias = [0 for _ in xrange(neuron[1])]
bias_2 = [0 for _ in xrange(neuron[2])]

for i in xrange(neuron[0]):
    for j in xrange(neuron[1]):
        weight[i][j] = 2*random.random()-1

for i in xrange(neuron[1]):
    for j in xrange(neuron[2]):
        weight_2[i][j] = 2*random.random()-1


for _ in xrange(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])]=1

        # Cost function, Square Root Eror
        eror = 0
        for i in xrange(3):
            eror +=  0.5*(target[i] - X_2[i])**2 
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in xrange(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j]*(1-X_2[j]))

        for i in xrange(neuron[1]):
            for j in xrange(neuron[2]):
                weight_2[i][j] -= alfa*(delta_2[j] * X_1[i])
                bias_2[j] -= alfa*delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in xrange(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j]*(1-X_1[j]))
        
        for i in xrange(neuron[0]):
            for j in xrange(neuron[1]):
                weight[i][j] -=  alfa*(delta_1[j] * x[i])
                bias[j] -= alfa*delta_1[j]
    
    cost_total /= len(train_X)
    if(_%100==0):
        print cost_total # Print cost untuk memantau training

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print preds

# Calculate accuration
acc = 0.0
for i in xrange(len(preds)):
    if preds[i]==int(test_y[i]):
        acc += 1
print acc / len(preds)*100, "%"