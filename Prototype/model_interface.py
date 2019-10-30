# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import model_from_json
import random
import numpy as np
import pytest

dash = "-" * 10

def load_model():
	loaded_model = None
	dataset = None
	varnum = None

	print("Choose model")
	print("1. Diabetes 2. Cancer")
	option = int(input("Choose: "))

	if(option == 1):
		dataset = loadtxt('models/datasets/pima_indians_diabetes.csv', delimiter=',')
		varnum = 8

		print("Loading model...")
		json_file = open('models/diab_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("models/diab_model.h5")
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	elif(option == 2):
		dataset = loadtxt('models/datasets/breast_cancer.csv', delimiter=',')
		varnum = 9

		print("Loading model...")
		json_file = open('models/cancer_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("models/cancer_model.h5")
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print("Loaded model from disk")
	return loaded_model, dataset, varnum


# evaluate the keras model
def evaluate(model):
	_, accuracy = model.evaluate(inputData, output)
	print('Accuracy: %.2f %%' % (accuracy*100))

#For testing
def predictClasses():
	# make probability predictions with the model
	# Accuracy increased by training with larger dataset, more hidden layers or adjusting epoch/batch_size
	rows = sum(1 for row in dataset)
	samples = int(input("Enter number of samples(Max %d): "% rows))

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
	if(varnum == 8):
		print("%25s %10s"% ("No. Times Pregnant: ", 		str(randinput[0])))
		print("%25s %10s"% ("Glucose Concentration : ", 	str(randinput[1])))
		print("%25s %10s"% ("Blood Pressure: ",				str(randinput[2])))
		print("%25s %10s"% ("Skin fold thickness: ",		str(randinput[3])))
		print("%25s %10s"% ("Serum Inculin: ",				str(randinput[4])))
		print("%25s %10s"% ("BMI: ",						str(randinput[5])))
		print("%25s %10s"% ("Diabetes Pedigree Func: ",		str(randinput[6])))
	elif(varnum == 9):
		print("%25s %10s"% ("Age: ", 		str(randinput[0])))
		print("%25s %10s"% ("BMI: ", 		str(randinput[1])))
		print("%25s %10s"% ("Glucose: ",	str(randinput[2])))
		print("%25s %10s"% ("Insulin: ",	str(randinput[3])))
		print("%25s %10s"% ("HOMA: ",		str(randinput[4])))
		print("%25s %10s"% ("Leptin: ",		str(randinput[5])))
		print("%25s %10s"% ("Adiponectin: ",str(randinput[6])))
		print("%25s %10s"% ("Resistin: ",	str(randinput[7])))
		print("%25s %10s"% ("MCP: ",		str(randinput[8])))

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

model, dataset, varnum = load_model()
inputData = dataset[:,0:varnum]
output = dataset[:,varnum]

displayMenu()

