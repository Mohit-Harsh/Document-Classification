from flask import Flask,request
import json
import pandas as pd
import numpy as np
from tensorflow import keras
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('../document_classifier.h5')


@app.route('/predict', methods=['Post'])
def predict():
    data = request.get_json(force=True)
    img = np.array(data['img'])
    prediction = int(np.round(model.predict(np.array([img]))))

    if prediction == 0:
        return "Aadhar Card"
    else:
        return "Pan Card"


if __name__ == "__main__":
    app.run()