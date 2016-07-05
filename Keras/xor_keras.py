from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# simple xor example
x = [[1,1],[1,0],[0,1],[0,0]]
y = [0, 1, 1, 0]

# build model with 1 hidden layer 4 neuron
model = Sequential()
model.add(Dense(output_dim=4, input_dim=2))
model.add(Activation("tanh"))
model.add(Dense(output_dim=1))
model.add(Activation("sigmoid"))

#change the default learning_rate
sgd = SGD(lr=0.05)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x, y, nb_epoch=300, batch_size=4)

loss = model.evaluate(x, y, batch_size=4)
classes = model.predict_classes(x, batch_size=4)

print loss
print classes