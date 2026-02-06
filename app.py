from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('vegetable_model.h5')

# Define labels based on your Dataset folders
labels = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 
          'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
          'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['file']
    img_path = "static/" + img_file.filename
    if not os.path.exists('static'):
        os.makedirs('static')
    img_file.save(img_path)

    # 🔥 MUST match training size
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = labels[np.argmax(prediction)]

    return f"<h2>Result: {result}</h2><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)


    