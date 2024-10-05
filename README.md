# Simply-CNN-to-CIFAR-10-Classifying
This repository contains a simple Convolutional Neural Network (CNN) using PyTorch to classify CIFAR-10 images. It features data augmentation for enhanced performance, logging of training and evaluation metrics, and visualizations of loss and accuracy to assess model performance.



## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

but in this case, I limited the dataset used to only 10.000 to save time on the training process.

## Data Augmentation
To improve the model's performance, the following data augmentation techniques are applied:

Random horizontal flip
Random crop with padding
Color jitter (brightness, contrast, saturation, and hue)
Model Architecture

## The CNN model defined in this project consists of the following layers:

### Convolutional Layer 1 :
Input channels: 3 (RGB)
Output channels: 16
Kernel size: 3x3
Activation: ReLU
Max Pooling Layer:
Kernel size: 2x2
Stride: 2
###Convolutional Layer 2 :
Input channels: 16
Output channels: 32
Kernel size: 3x3
Activation: ReLU
Max Pooling Layer:
Kernel size: 2x2
Stride: 2
### Fully Connected Layer 1 :
Input features: 32 * 8 * 8 (flattened)
Output features: 128
Activation: ReLU
### Fully Connected Layer 2 :
Input features: 128
Output features: 10 (number of classes)
Training

## The model is trained for 20 epochs with the Adam optimizer and Cross Entropy Loss function. During training, the following metrics are logged:

Training loss
Training accuracy
Testing loss
Testing accuracy
Evaluation

## After training, the model's performance is evaluated using the test dataset. The evaluation includes:

Loss and accuracy plots for both training and testing phases.
Confusion matrix to visualize the performance across different classes.
Usage

## To run the training process, simply execute the following command in your terminal:

bash
Salin kode
python cnn_model.py
Make sure to adjust the code if necessary, based on your environment setup.

## Conclusion

This project demonstrates the implementation of a simple CNN model for image classification tasks. The usage of data augmentation techniques and evaluation metrics helps in understanding the model's performance better.
