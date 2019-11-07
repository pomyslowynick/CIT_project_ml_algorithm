import json
from keras.models import Sequential
from keras.layers import Dense
import random
import boto3

def lambda_handler(event, payload):
	# get model and inputs from payload

	# call predict class with model and inputs
	model = 1
	inputs = [6,148,72,35,0,33.6,0.627,50]
	
	prediction = predictClass(model, inputs)
	
	
	return {
        'statusCode': 200,
        'body': json.dumps(object_key)
    }


def predictClass(inmodel, inputs):
	# load correct model
	model = load_model(inmodel)

	# add inputs to np array
	test = np.array([inputs])
	# test - numpy array with vars
	prediction = model.predict(test)

	# add this to json and return prediction[0]*100
	return int(prediction[0]*100)

	# add patient details to database
	
	


def load_model(model):
	loaded_model = None
	s3 = boto3.client('s3')

	if(model == 1):
		json_file = s3.get_object(Bucket='group-project-cit', Key='diab_model.json')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		weights = s3.get_object(Bucket='group-project-cit', Key='diab_model.h5')
		loaded_model.load_weights(weights)
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


