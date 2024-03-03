# Lung Disease Classification using Machine Learning

This project aims to classify lung diseases (normal, COVID-19, tuberculosis, pneumonia) using machine learning techniques. The model is trained on a dataset containing X-ray images of lungs.

## Dataset

The dataset used for training the model consists of X-ray images of lungs, categorized into four classes: normal, COVID-19, tuberculosis, and pneumonia. Each class contains a large number of images for training the model effectively.

## Model Training

The model is trained using TensorFlow and Keras. The following steps were followed for training:

1. Preprocessing: X-ray images are loaded and preprocessed using image preprocessing techniques such as resizing, normalization, and augmentation.

2. Label Encoding: Labels for the images are encoded using one-hot encoding to represent the different classes (normal, COVID-19, tuberculosis, pneumonia).

3. Model Architecture: The model architecture consists of convolutional neural network (CNN) layers followed by fully connected layers to learn features from the images and classify them into respective classes.

4. Training: The model is trained using the preprocessed images and their corresponding labels. The dataset is split into training and validation sets to evaluate the model's performance during training.

5. Evaluation: The trained model's performance is evaluated on a separate test set to assess its accuracy and effectiveness in classifying lung diseases.

## Dependencies

- Python (>=3.6)
- TensorFlow
- scikit-learn
- NumPy
