from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
dataset = loadtxt('pima_indians_diabetes.csv', delimiter=',')

# split into input (inputData) and output (output) variables
# inputData - input variables
# Y - Output 0 and 1
inputData = dataset[:,0:8]
output = dataset[:,8]

# define the keras model (hidden layers)
# Accuracy increased from 70% to 90% by adding 2 more hiddden layers
# Nodes increased in second hidden layer to improve acuracy
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# epoch = complete pass through all rows
# batch_size = Samples considered by the model within an epoch before weights are updated.
model.fit(inputData, output, epochs=1000, batch_size=10, verbose=0)

model_json = model.to_json()

with open("diab_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
_, accuracy = model.evaluate(inputData, output)
print('Accuracy: %.2f %%' % (accuracy * 100))


