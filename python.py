from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
from flask import *
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)
g_model = None

# Classes of meters
classes = { 0:'Real',
            1:'Spoof', 
            }
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def image_processing(img):
    global g_model
    if g_model is None:
        g_model = load_model('Knight.h5')
        print("model init")
    data=[]
    image = Image.open(img)
    image = image.resize((100,100))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = g_model.predict_classes(X_test)
    print(Y_pred)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return result
    elif request.method == 'GET' and request.args.get('url', default=None):
        url = request.args.get('url', default=None)
        r = requests.get(url, allow_redirects=True)
        filename = str(time.time()) + '.jpg'
        print(filename)
        open(filename, 'wb').write(r.content)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    g_model = load_model('Knight.h5')
    app.run(host='0.0.0.0',port=80,debug=True)
