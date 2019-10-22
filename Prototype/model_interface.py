# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import model_from_json
import random
import numpy as np
import pytest

dataset = loadtxt('pima_indians_diabetes.csv', delimiter=',')
dash = "-" * 10
inputData = dataset[:,0:8]
output = dataset[:,8]

def load_model():
	print("Loading model...")
	json_file = open('diab_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print("Loaded model from disk")

	return loaded_model


# evaluate the keras model
def evaluate(model):
	_, accuracy = model.evaluate(inputData, output)
	print('Accuracy: %.2f %%' % (accuracy*100))

#For testing
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

#For testing
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
	print('Risk: %d %%'%  (prediction[0]*100))
	print("Actual Output: " + str(randoutput))
	print(dash)
	#Actual Output: 1 - Diabetic 0 - Not Diabetic
	#Prediction - Percent chance inputs will return positive for diabetes

def testClass(x):
	inputs = inputData.tolist()
	input = inputs[x]

	test = np.array([input])
	prediction = model.predict(test)

	return float(prediction[0])



def displayMenu():
	run = True
	while(run):
		print("--------------------------------------")
		print("1. Evaluate the model")
		print("2. See sample predictions")
		print("3. Get prediction for random input")
		print("4. Pytest output")
		print("5. Exit")

		option = int(input("Enter 1-5: "))
		if(option == 1):
			evaluate(model)
		elif(option == 2):
			predictClasses()
		elif(option == 3):
			predictClass()
		elif (option == 4):
			test_Diab()
		elif(option == 5):
			run = False

def test_Diab():
	assert testClass(0) == 0.83139568567276


model = load_model()

displayMenu()

