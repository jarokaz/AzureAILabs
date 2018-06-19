import json
import numpy
import tensorflow as tf
from azureml.assets.persistence.persistence import get_model_path
from tensorflow.python.keras.models import load_model

def init():
    global model
    
    model_path = 'best_model.h5'
    # deserialize the model file back into a sklearn model
    model = load_model(model_path)

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})