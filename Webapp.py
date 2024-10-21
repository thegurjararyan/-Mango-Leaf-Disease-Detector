from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec/'

# Define the upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # Limit file size to 5 MB

# Load the trained model
model = load_model('mangodisease.h5')
disease_classes = {
    0: 'Anthracnose',
    1: 'Cutting Weevil',
    2: 'Die Back',
    3: 'Healthy',
    4: 'Powdery Mildew',
}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess images
def preprocess_image(img_path, target_size=(256, 256)):
    # Read the image and convert to RGB
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image and normalize pixel values
    img = cv2.resize(img, target_size)
    img = np.array(img) / 255.0
    # Add batch dimension for model input
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the class and confidence
def model_predict(img_path, model):
    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)
    # Make predictions using the model
    predictions = model.predict(preprocessed_img)
    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions)
    # Calculate the confidence level (percentage)
    confidence_percentage = np.max(predictions) * 100
    return predicted_class_index, confidence_percentage, predictions[0]

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    # Get the uploaded file
    file = request.files['file']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file type is allowed and size is within limit
    if file and allowed_file(file.filename) and len(file.read()) <= MAX_FILE_SIZE:
        file.seek(0)  # Reset file pointer after reading

        # Secure the file name
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Try to predict the class and confidence level
        try:
            predicted_class_index, confidence_percentage, probabilities = model_predict(file_path, model)
            predicted_class = disease_classes[predicted_class_index]

            # Remove the file after prediction
            os.remove(file_path)

            # Return the prediction probabilities and class labels
            return jsonify({
                'message': f"{predicted_class} with {confidence_percentage:.2f}% confidence",
                'probabilities': {
                    'Anthracnose': probabilities[0] * 100,
                    'Cutting Weevil': probabilities[1] * 100,
                    'Die Back': probabilities[2] * 100,
                    'Healthy': probabilities[3] * 100,
                    'Powdery Mildew': probabilities[4] * 100,
                }
            })
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'Error during prediction. Please try again.'}), 500

    else:
        return jsonify({'error': 'Invalid file type or file size exceeds limit.'}), 400

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
