# Image Classification Project

A comprehensive deep learning project for image classification using Convolutional Neural Networks (CNN) implemented in Google Colab.

## Project Overview

This project implements an end-to-end image classification pipeline with data preprocessing, augmentation, model training, and evaluation. The model classifies images into 5 different categories using a custom CNN architecture.

## Features

- **Exploratory Data Analysis (EDA)** - Comprehensive analysis of the image dataset
- **Data Preprocessing** - Automated train/test split with organized directory structure
- **Data Augmentation** - Image transformations to prevent overfitting
- **Deep Learning Model** - Custom CNN architecture with multiple convolutional layers
- **Performance Monitoring** - Early stopping and model evaluation metrics

## Dataset Structure

The project automatically organizes the dataset with an 85/15 train/test split:

```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   ├── class_3/
│   ├── class_4/
│   └── class_5/
└── test/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    ├── class_4/
    └── class_5/
```

## Data Analysis & Preprocessing

### Exploratory Data Analysis
- Statistical analysis of image distribution across classes
- Visualization of dataset composition using bar plots
- Image count verification for balanced dataset assessment

### Data Augmentation
The project implements several augmentation techniques to enhance model generalization:
- **Rescaling**: Normalizes pixel values to [0,1] range
- **Resizing**: Standardizes image dimensions for consistent input
- **Rotation**: Introduces rotational variance to prevent overfitting
- **Visualization**: Displays augmented images for verification

## Model Architecture

### CNN Model Specifications
- **Input Shape**: (img_height, img_width, 3)
- **Convolutional Layers**: Progressive feature extraction with increasing filter sizes
- **Pooling Layers**: MaxPooling for spatial dimension reduction
- **Dense Layers**: Fully connected layers for classification
- **Output**: 5 classes with softmax activation

### Architecture Details
```python
Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')
])
```

## Training Configuration

### Model Compilation
- **Loss Function**: Categorical Crossentropy (measures difference between true and predicted distributions)
- **Optimizer**: Adam (adaptive learning rate optimization)
- **Activation Functions**: ReLU for non-linearity introduction

### Training Features
- **Early Stopping**: Halts training after 5 consecutive epochs without improvement
- **Model Checkpointing**: Saves best model with verbose output
- **Performance Monitoring**: Real-time tracking of training metrics

## Model Performance

### Confusion Matrix Results
```
Predicted →  [0] [1] [2] [3] [4]
Actual ↓
[0]          [ 0  2  0  2  0]
[1]          [ 4  3  1  1  1]
[2]          [ 0  0  6  0  0]
[3]          [ 0  0  0  3  0]
[4]          [ 0  0  1  0  6]
```

### Performance Analysis
- **Class 2**: Perfect classification (6/6 correct)
- **Class 3**: Excellent performance (3/3 correct)
- **Class 4**: Strong performance (6/7 correct)
- **Class 1**: Moderate performance with some misclassification
- **Class 0**: Challenging class with classification difficulties

## Usage Instructions

1. **Environment Setup**: Open the notebook in Google Colab
2. **Data Upload**: Upload your image dataset to Colab
3. **Path Configuration**: Specify directory paths for your dataset
4. **Execution**: Run cells sequentially for complete pipeline execution
5. **Evaluation**: Review confusion matrix and performance metrics

## Key Components Explained

### MaxPooling2D
Reduces spatial dimensions by selecting maximum values from 2x2 regions, helping to:
- Reduce computational complexity
- Extract dominant features
- Provide translation invariance

### Categorical Crossentropy
Measures the difference between actual and predicted probability distributions, ideal for multi-class classification problems.

### Adam Optimizer
Adaptive learning rate algorithm that combines advantages of AdaGrad and RMSProp for efficient model optimization.

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Google Colab environment

## Future Improvements

- Implement transfer learning with pre-trained models
- Add more sophisticated data augmentation techniques
- Experiment with different architectures (ResNet, VGG, etc.)
- Include cross-validation for robust performance evaluation
- Add model interpretability features
