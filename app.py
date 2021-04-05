import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from flask import Flask, jsonify, request, url_for
from flask_cors import CORS, cross_origin

app = Flask(__name__, static_url_path='/output', static_folder='output')  
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = "E:\\CaptUploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "model"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer':hub.KerasLayer})

dataset_labels = np.array(['Document', 'Garbage'])

@app.route('/upload', methods = ['POST'])  
@cross_origin()
def success():  
    if request.method == 'POST':  
        f = request.files['file']

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        img = tf.keras.preprocessing.image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), target_size = (224, 224))
        img = np.expand_dims(img, axis = 0)
        img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img)

        tf_model_predictions = model.predict(img_preprocessed)
        predicted_ids = np.argmax(tf_model_predictions, axis=-1)
        predicted_labels = dataset_labels[predicted_ids]

        results = {'result': predicted_labels[0]}

        return jsonify(results)

if __name__ == '__main__':  
    app.run(debug = False)  