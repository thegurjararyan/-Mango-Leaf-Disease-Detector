# Mango Leaf Disease Detector

[![License](https://img.shields.io/github/license/thegurjararyan/-Mango-Leaf-Disease-Detector)](https://github.com/thegurjararyan/-Mango-Leaf-Disease-Detector/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/thegurjararyan/-Mango-Leaf-Disease-Detector)](https://github.com/thegurjararyan/-Mango-Leaf-Disease-Detector/issues)
[![Forks](https://img.shields.io/github/forks/thegurjararyan/-Mango-Leaf-Disease-Detector)](https://github.com/thegurjararyan/-Mango-Leaf-Disease-Detector/network/members)
[![Stars](https://img.shields.io/github/stars/thegurjararyan/-Mango-Leaf-Disease-Detector)](https://github.com/thegurjararyan/-Mango-Leaf-Disease-Detector/stargazers)

## Project Description

The **Mango Leaf Disease Detector** is a web application designed to assist farmers and agriculturists in diagnosing diseases in mango leaves using machine learning. The model predicts several common mango leaf diseases, including Anthracnose, Cutting Weevil, Die Back, Powdery Mildew, and identifies healthy leaves. The goal is to provide users with a simple yet powerful tool to identify mango leaf diseases, along with the symptoms, causes, and prevention methods for each disease.

The trained model uses **Convolutional Neural Networks (CNNs)** and is built for optimal accuracy and performance. The current model file (`mangodisease.h5`) is 97 MB and can classify images of mango leaves into specific disease categories.

## Features

- **Disease Detection:** Identifies diseases such as Anthracnose, Cutting Weevil, Die Back, Powdery Mildew, and healthy leaves.
- **Prevention Methods:** Provides details on prevention strategies for each disease.
- **User-Friendly Interface:** Simple image upload page with enhanced visuals, including funny quotes for a friendly experience.

## Libraries and Technologies Used

- **Machine Learning/Deep Learning:**
  - TensorFlow (Keras, MobileNetV2, Conv2D, MaxPooling2D, Dropout, etc.)
  - scikit-learn
- **Data Handling & Visualization:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
- **Image Processing:**
  - OpenCV
  - Pillow
- **Web Technologies:**
  - HTML/CSS (for web app design)
  - Flask (for web framework)

## Dataset

The model was trained on a curated dataset of mango leaves, each labeled with their corresponding disease. The dataset includes both healthy leaves and leaves affected by diseases such as:
- **Anthracnose**
- **Cutting Weevil**
- **Die Back**
- **Powdery Mildew**
- **Healthy**

Images were pre-processed, resized, and augmented using TensorFlow's `ImageDataGenerator` to improve the model’s robustness.

## Model Performance

The model has been trained and tested on a dataset of mango leaf images. The architecture is based on **MobileNetV2**, which has been fine-tuned to classify diseases in mango leaves with a high level of accuracy.

## How to Run the Application

1. **Clone the repository:**

   ```bash
   git clone https://github.com/thegurjararyan/-Mango-Leaf-Disease-Detector.git

Navigate to the project directory:

bash
Copy code
cd -Mango-Leaf-Disease-Detector
Install the required dependencies:

Make sure you have Python installed, then install the required libraries using pip:

bash
Copy code
pip install -r requirements.txt
Run the web application:

Start the Flask web server:

bash
Copy code
python app.py
Access the web app:

Open your browser and go to http://localhost:5000. Upload an image of a mango leaf to receive a prediction.

Web Application Demo
A live demo of the web app is hosted at: Mango Disease Detector

Contributing
If you'd like to contribute to this project, feel free to open a pull request or raise an issue. Contributions, issues, and feature requests are always welcome!

Fork the repository
Create your feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a new Pull Request
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or suggestions, feel free to reach out to:

Aryan Chaudhary: Twitter @thegurjararyan
   
