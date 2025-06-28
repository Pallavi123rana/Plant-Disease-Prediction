

import os
import json
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/test_images'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'plant_disease_prediction_model.h5')
model = tf.keras.models.load_model(model_path)

class_indices_path = os.path.join(working_dir, 'class_indices.json')
class_indices = json.load(open(class_indices_path))


# Load & Preprocess Image Function (Pillow)
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


# Predict Function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict class using Pillow logic
        prediction = predict_image_class(model, filepath, class_indices)

        image_url = f"/static/test_images/{file.filename}"
        return render_template("predict.html", prediction=prediction, image=image_url)

    return render_template("predict.html", prediction=None)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Message from {name} ({email}): {message}")
        return render_template('contact.html', success=True)
    return render_template('contact.html', success=False)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "1234":
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html', error=None)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)


