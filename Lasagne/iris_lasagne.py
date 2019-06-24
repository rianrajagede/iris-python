from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training

the datasets has separated to two file from originai datasets:
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
learning rate = 0.01
epoch = 500
"""
import theano
import theano.tensor as T
import numpy as np
import lasagne

#initiate theano variable
input_val = T.fmatrix("inputs")
target_val = T.ivector("targets")

#build model
input_layer  = lasagne.layers.InputLayer(shape=xtrain.shape, input_var=input_val)
hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=10,nonlinearity=lasagne.nonlinearities.rectify)   
output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=3,nonlinearity=lasagne.nonlinearities.softmax)   
output_val =  output_layer.get_output()

#choose objective/loss function 
objective = lasagne.objectives.Objective(
                output_layer,
                loss_function=lasagne.objectives.categorical_crossentropy)                
loss = objective.get_loss(target=target_val)

#choose optimizer
all_params = lasagne.layers.get_all_params(output_layer)
updates = lasagne.updates.sgd(loss, all_params, learning_rate=0.01)

#compile theano function
train_model = theano.function([input_val,target_val],loss,allow_input_downcast=True,updates=updates)
test_model = theano.function([input_val],output_val,allow_input_downcast=True)

#train
for _ in range(500):   
    loss_val = train_model(xtrain,ytrain)
    prediction = np.argmax(test_model(xtrain),axis=1)
    accuration = 100*np.mean(ytrain == prediction)
    print("Epoch " + str(_+1) + "/" + str(500) + " - loss: " + str(loss_val) + " - accuration: " + str(accuration))
    
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
xtest= datatest_array[:,:4]
ytest = datatest_array[:,4]

#get prediction
prediction = np.argmax(test_model(xtest),axis=1)

#get accuration
accuration = 100*np.mean(ytest == prediction)
print("Test Accuration : "+str(accuration))
print("Prediction :")
print(prediction)
print("Target :")
print(np.asarray(ytest,dtype="int32"))