# load Flask 
import flask
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import numpy as np

app = flask.Flask(__name__)
# define a predict function as an endpoint 
@app.route("/", methods=["GET","POST"])

def getPrediction():
    values = []
    keys = []
    
    # check for passed in parameters   
    params = flask.request.json
    if params is None:
        params = flask.request.args
    
    # if parameters are found, echo the msg parameter 
    if "model" in params.keys(): 
        model = int(params["model"])
        for key,value in params.items():
            if(key != "model"):
                values.append(value)
    
    result = predictClass(model, values)
    print(result)
    # return a response in json format 
    return flask.jsonify(result)

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
    if(model == 1):
        json_file = open('models/diab_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("models/diab_model.h5")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        loaded_model.save("d_model")

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
    else:
        print("No model loaded")
    
    
    return loaded_model

# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0')


