"""
SECTION 1 : Load and setup data for training
"""
import pandas as pd

#load
datatrain = pd.read_csv('iris/iris_train.csv')

#change string value to numeric
datatrain.set_value(datatrain['species']=='Iris-setosa',['species'],0)
datatrain.set_value(datatrain['species']=='Iris-versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='Iris-virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.as_matrix()

#split x and y (feature and target)
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]

"""
SECTION 2 : Build model
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

#change target format
ytrain = np_utils.to_categorical(ytrain) 

#model with 1 hidden layer 10 neuron, 500 epoch, no batch, gradient descent
model = Sequential()
model.add(Dense(output_dim=10, input_dim=4))
model.add(Activation("relu"))
model.add(Dense(output_dim=3))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(xtrain, ytrain, nb_epoch=500, batch_size=120)

"""
SECTION 3 : Testing model
"""
#load
datatest = pd.read_csv('iris/iris_test.csv')

#change string value to numeric
datatest.set_value(datatest['species']=='Iris-setosa',['species'],0)
datatest.set_value(datatest['species']=='Iris-versicolor',['species'],1)
datatest.set_value(datatest['species']=='Iris-virginica',['species'],2)
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatest_array = datatest.as_matrix()

#split x and y (feature and target)
xtest= datatest_array[:,:4]
ytest = datatest_array[:,4]

#get prediction
classes = model.predict_classes(xtest, batch_size=120)

"""
SECTION 4 : Get Accuration
"""
import numpy as np
accuration = np.sum(classes == ytest)/30.0 * 100

print classes
print ytest
print str(accuration) + '%'