"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data

SECTION 2 : Build and Train Model
    
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation function using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = default from keras.optimizer.SGD, 0.01
epoch = 500

SECTION 3 : Testing model
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

def main():
    
    """
    SECTION 1 : Load and setup data for training
    """
    
    # Load the datatrain
    datatrain = pd.read_csv('../Datasets/iris/iris_train.csv')
    
    # Change string value to numeric
    datatrain.set_value(datatrain['species'] == 'Iris-setosa', ['species'], 0)
    datatrain.set_value(datatrain['species'] == 'Iris-versicolor', ['species'], 1)
    datatrain.set_value(datatrain['species'] == 'Iris-virginica', ['species'], 2)
    datatrain = datatrain.apply(pd.to_numeric)
    
    # Change dataframe to array
    datatrain_array = datatrain.as_matrix()
    
    # Split x and y (feature and target)
    xtrain = datatrain_array[:, :4]
    ytrain = datatrain_array[:, 4]
    
    """
    SECTION 2 : Build and Train Model
    """
    
    # Change target format
    ytrain = np_utils.to_categorical(ytrain) 
    
    # Build model
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=4))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=3))
    model.add(Activation("softmax"))
    
    # Choose optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    
    # Start training
    model.fit(xtrain, ytrain, nb_epoch=500, batch_size=120)

    """
    SECTION 3 : Testing model
    """
    
    # Load the datatest
    datatest = pd.read_csv('../Datasets/iris/iris_test.csv')
    
    # Change string value to numeric
    datatest.set_value(datatest['species'] == 'Iris-setosa', ['species'], 0)
    datatest.set_value(datatest['species'] == 'Iris-versicolor', ['species'], 1)
    datatest.set_value(datatest['species'] == 'Iris-virginica', ['species'], 2)
    datatest = datatest.apply(pd.to_numeric)
    
    # Change dataframe to array
    datatest_array = datatest.as_matrix()
    
    # Split x and y (feature and target)
    xtest = datatest_array[:, :4]
    ytest = datatest_array[:, 4]
    
    # Get prediction
    classes = model.predict_classes(xtest, batch_size=120)
    
    # Get accuration
    accuration = np.sum(classes == ytest)/30.0 * 100

    print "Test Accuration : " + str(accuration) + '%'
    print "Prediction :"
    print classes
    print "Target :"
    print np.asarray(ytest,dtype="int32")

if __name__ == "__main__":
    main()