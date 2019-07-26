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
output layer : 3 neuron, represents the class of Iris, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = default from keras.optimizer.SGD, 0.01
epoch = 500
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical

#change target format
ytrain = to_categorical(ytrain) 

#build model
model = tf.keras.Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Activation("relu"))
model.add(Dense(3))
model.add(Activation("softmax"))

#choose optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#train
model.fit(xtrain, ytrain, epochs=500, batch_size=32)

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
classes = model.predict_classes(xtest, batch_size=32)

#get accuration
import numpy as np
accuration = np.sum(classes == ytest)/len(ytest) * 100

print("Test Accuration : " + str(accuration) + '%')
print("Prediction :")
print(classes)
print("Target :")
print(np.asarray(ytest,dtype="int32"))