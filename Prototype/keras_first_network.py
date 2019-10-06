# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np

model = Sequential()
dataset = loadtxt('pima_indians_diabetes.csv', delimiter=',')
dash = "-" * 10

def initAlg():
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

	return inputData, output

# evaluate the keras model
def evaluate():
	_, accuracy = model.evaluate(inputData, output)
	print('Accuracy: %.2f %%' % (accuracy*100))


def predictClasses():
	# make probability predictions with the model
	# Accuracy increased by training with larger dataset, more hidden layers or adjusting epoch/batch_size
	samples = int(input("Enter number of samples(Max 768): "))

	predictions = model.predict(inputData)
	for i in range(samples):
		if(predictions[i].round() != output[i]):
			print('%s => %.2f (expected %d)' % (inputData[i].tolist(), predictions[i], output[i]) + " X")
		else:
			print('%s => %.2f (expected %d)' % (inputData[i].tolist(), predictions[i], output[i]))


def predictClass():
	rows = sum(1 for row in dataset)
	rand = random.randint(0, rows)

	inputs = inputData.tolist()
	randinput = inputs[rand]
	randoutput = output[rand]

	test = np.array([randinput])
	prediction = model.predict(test)

	print(dash + "Details" + dash)
	print("%25s %10s"% ("No. Times Pregnant: ", 		str(randinput[0])))
	print("%25s %10s"% ("Glucose Concentration : ", 	str(randinput[1])))
	print("%25s %10s"% ("Blood Pressure: ",				str(randinput[2])))
	print("%25s %10s"% ("Skin fold thickness: ",		str(randinput[3])))
	print("%25s %10s"% ("Serum Inculin: ",				str(randinput[4])))
	print("%25s %10s"% ("BMI: ",						str(randinput[5])))
	print("%25s %10s"% ("Diabetes Pedigree Func: ",		str(randinput[6])))
	print(dash)
	print('Prediction(%%): %.2f %%'%  prediction[0])
	print("Actual Output: " + str(randoutput))
	print(dash)
	#Actual Output: 1 - Diabetic 0 - Not Diabetic
	#Prediction - Percent chance inputs will return positive for diabetes

def displayMenu():
	run = True
	while(run):
		print("--------------------------------------")
		print("1. Evaluate the model")
		print("2. See sample predictions")
		print("3. Get prediction for random input")
		print("4. Exit")

		option = int(input("Enter 1-4: "))
		if(option == 1):
			evaluate()
		elif(option == 2):
			predictClasses()
		elif(option == 3):
			predictClass()
		elif(option == 4):
			run = False

print("Training...")
inputData, output = initAlg()
displayMenu()