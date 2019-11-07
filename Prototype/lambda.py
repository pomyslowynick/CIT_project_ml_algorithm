from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from array import array

	# get model and inputs from json

	# call predict class with model and inputs

	
	


def predictClass(inmodel, inputs):
	# load correct model
	model = load_model(inmodel)
	prediction = model.predict(np.array([inputs]))

	# add this to json and return prediction[0]*100
	return int(prediction[0]*100)

	# add patient details to database

def load_model(model):
	loaded_model = None
	if(model == 1):
		json_file = open('models/diab_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("models/diab_model.h5")
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	elif(model == 2):
		json_file = open('models/cancer_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("models/cancer_model.h5")
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	elif(model == 3):
		json_file = open('models/heartd_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("models/heartd_model.h5")
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return loaded_model

model = 1
inputs = [6,148,72,35,0,33.6,0.627,50]

prediction = predictClass(model, array)

print(prediction)
