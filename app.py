import base64
import io
import os
from imageio import imread
import numpy as np
import tensorflow as tf
from flask import Flask, request

# Image preprocessing function
preprocessing_function = tf.keras.applications.resnet50.preprocess_input

# ID to class
id2class = {0: 'Cheese Cake', 1: 'Donuts', 2: 'Dumplings', 3: 'Fried Rice', 4: 'Burger', 5: 'Omelette', 6: 'Pancakes', 7: 'Pizza', 8: 'Steak', 9: 'Tacos'}
id2meal = {0: 'cheesecake', 1: 'donut', 2: 'dumpling', 3: 'rice', 4: 'burger', 5: 'omelette', 6: 'pancake', 7: 'pizza', 8: 'steak', 9: 'taco'}

# Load model
print('Loading model...')
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(256, 256, 3), pooling='avg')
output = base_model.output
output = tf.keras.layers.Flatten()(output)
output = tf.keras.layers.Dropout(0.25)(output)
output = tf.keras.layers.Dense(128, activation='relu')(output)
output = tf.keras.layers.Dropout(0.25)(output)
output = tf.keras.layers.Dense(32, activation='relu')(output)
output = tf.keras.layers.Dense(10, activation='softmax')(output)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Load weights
print('Loading weights')
model.load_weights('./weights.h5')

# Preproces image
def preprocess_image(image_base64):
    print('Preprocessing image...')
    image = imread(io.BytesIO(base64.b64decode(image_base64)))
    image = tf.image.resize([image], (256, 256))[0].numpy().astype(int)
    image = preprocessing_function(image)
    return image

# Sample inference
def infer(image_base64, model):
    image = preprocess_image(image_base64)
    image = np.expand_dims(image, axis=0)
    print('Input shape: ', image.shape)
    category_arr = model.predict(image)[0]
    category_id = np.argmax(category_arr)
    category_name = id2class[category_id]
    category_meal = id2meal[category_id]
    return category_name, category_meal, category_arr[category_id]

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return 'Connection Successfull!'

@app.route('/predict', methods=['POST'])
def predict():
    image_base64 = request.get_json(force=True)['image']
    if image_base64 is None:
        return {'success': False, 'category': 'NULL'}

    category, meal, confidence = infer(image_base64, model)

    return {'success': True, 'category': category, 'meal': meal, 'confidence': str(confidence)}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3001)))
