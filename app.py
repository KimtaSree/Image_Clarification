from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Pre-trained Model (e.g., MobileNet)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Resize image to match model's expected input size
    image_array = np.array(image)
    if image_array.shape[-1] != 3:  # Ensure image has 3 channels (RGB)
        raise ValueError("Image does not have 3 channels (RGB)")
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

def predict(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
    return decoded_predictions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                predictions = predict(filepath)
                return render_template('result.html', predictions=predictions, filename=filename)
            except Exception as e:
                # Handle errors during the image processing or model prediction
                return f"An error occurred: {e}", 500
        else:
            return "Invalid file type. Please upload a valid image.", 400
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
