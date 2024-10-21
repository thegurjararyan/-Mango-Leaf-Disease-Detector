# Mango Leaf Disease Detector



## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description

The **Mango Leaf Disease Detector** is an innovative web application designed to empower farmers and agriculturists with the ability to diagnose diseases in mango leaves using state-of-the-art machine learning techniques. By leveraging Convolutional Neural Networks (CNNs), our model accurately predicts several common mango leaf diseases, including:

- Anthracnose
- Cutting Weevil
- Die Back
- Powdery Mildew

Additionally, the system can identify healthy leaves, providing a comprehensive tool for mango crop management.

Our goal is to offer users a simple yet powerful solution for early disease detection, complemented by detailed information on symptoms, causes, and prevention methods for each identified disease.

## Features

- **Accurate Disease Detection:** Utilizes a CNN model to classify mango leaf images into specific disease categories with high precision.
- **Comprehensive Disease Information:** Provides detailed insights on symptoms, causes, and prevention strategies for each detected disease.
- **User-Friendly Interface:** Features an intuitive image upload system with enhanced visuals and engaging content for a seamless user experience.
- **Real-time Analysis:** Offers instant disease classification upon image upload.
- **Mobile-Friendly Design:** Ensures accessibility across various devices for field use.

## Technologies Used

### Machine Learning / Deep Learning
- TensorFlow 2.x
- Keras
- MobileNetV2 (pre-trained model)
- scikit-learn

### Data Handling & Visualization
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Image Processing
- OpenCV
- Pillow (PIL)

### Web Technologies
- Flask (Python web framework)
- HTML5
- CSS3
- JavaScript

## Dataset

Our model is trained on a meticulously curated dataset of mango leaf images, encompassing:

- Healthy leaves
- Leaves affected by Anthracnose
- Leaves damaged by Cutting Weevil
- Leaves showing signs of Die Back
- Leaves with Powdery Mildew

The dataset underwent rigorous preprocessing, including resizing and augmentation using TensorFlow's `ImageDataGenerator`, to enhance the model's robustness and generalization capabilities.

## Model Performance

The core of our system is a fine-tuned MobileNetV2 architecture, optimized for mango leaf disease classification. Key performance metrics include:

- **Accuracy:** 95% on the test set
- **Precision:** 94%
- **Recall:** 93%
- **F1-Score:** 93.5%

These metrics demonstrate the model's high reliability in disease detection across various conditions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thegurjararyan/Mango-Leaf-Disease-Detector.git
   cd Mango-Leaf-Disease-Detector
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask web server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Upload an image of a mango leaf using the provided interface.

4. Review the analysis results, including the detected disease (if any) and recommended prevention measures.

## Contributing

We welcome contributions from the community! If you'd like to improve the Mango Leaf Disease Detector, please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Aryan Chaudhary - [@thegurjararyan](https://twitter.com/thegurjararyan)

Project Link: [https://github.com/thegurjararyan/Mango-Leaf-Disease-Detector](https://github.com/thegurjararyan/Mango-Leaf-Disease-Detector)

---

Thank you for your interest in the Mango Leaf Disease Detector project. We're excited to see how this tool can make a difference in mango cultivation practices worldwide!
