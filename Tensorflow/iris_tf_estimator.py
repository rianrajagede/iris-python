from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training and testing

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""
import pandas as pd

#load data train
datatrain = pd.read_csv('Datasets/iris/iris_train.csv')

#change string value to numeric
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.values

#split x and y (feature and target)
xtrain = datatrain_array[:,:-1]
ytrain = datatrain_array[:,-1].astype(int)

#load data test
datatest = pd.read_csv('Datasets/iris/iris_test.csv')

#change string value to numeric
datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatest_array = datatest.values

#split x and y (feature and target)
xtest = datatest_array[:,:-1]
ytest = datatest_array[:,-1].astype(int)

"""
SECTION 2 : Train and evaluate a model

There are three steps before build a tf estimator models:
1. Define input function
2. Define feature columns
3. Define an estimator

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.001
epoch = 500
"""

import tensorflow as tf
num_epoch = 500
num_train = 120
num_test = 30

# 1 Define input function
def input_function(x, y, is_train):
    dict_x = {
        "sepal_length" : x[:,0],
        "sepal_width" :  x[:,1],
        "petal_length" : x[:,2],
        "petal_width" :  x[:,3]
    }

    dataset = tf.data.Dataset.from_tensor_slices((
        dict_x, y
    ))

    if is_train:
        # batch(num_train) or batch(num_test) means no batch
        dataset = dataset.shuffle(num_train).repeat(num_epoch).batch(num_train)
    else:
        dataset = dataset.batch(num_test)

    return dataset

def main(argv):
    # 2 Define feature columns
    feature_columns = [
        tf.feature_column.numeric_column(key="sepal_length"),
        tf.feature_column.numeric_column(key="sepal_width"),
        tf.feature_column.numeric_column(key="petal_length"),
        tf.feature_column.numeric_column(key="petal_width")
    ]

    # 3 Define an estimator
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10],
        n_classes=3,
        optimizer=tf.train.GradientDescentOptimizer(0.001),
        activation_fn=tf.nn.relu
    )

    # Train the model
    classifier.train(
        input_fn=lambda:input_function(xtrain, ytrain, True)
    )

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda:input_function(xtest, ytest, False)
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == "__main__":
    tf.set_random_seed(1103) # avoiding different result of random
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)