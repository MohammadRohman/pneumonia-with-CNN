# Pneumonia Classification with Custom CNN Layer

This project implements a Pneumonia Classification system using a Custom Convolutional Neural Network (CNN) designed for binary classification: distinguishing between Normal and Pneumonia-affected chest X-ray images. It leverages PyTorch for model development and training.

## Features 
 - Custom CNN Architecture: Includes convolutional, pooling, and fully connected layers optimized for chest X-ray classification.
 - Data Augmentation: Improves generalization with techniques such as random rotations and flips.
 - Multi-GPU Support: Automatically utilizes multiple GPUs if available.
 - Visualization: Includes plots for training history and confusion matrices.
 - Efficient Training: Utilizes mixed-precision training with PyTorch's autocast and GradScaler.

## Datasets
The dataset contains chest X-ray images categorized into two classes:

NORMAL: Healthy individuals.
PNEUMONIA: Individuals diagnosed with pneumonia.

you can download the datasets from here : https://www.kaggle.com/code/arbazkhan971/pneumonia-detection-using-cnn-96-accuracy

## Prequisites
install the required libraries
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn Pillow

## Project Structures 
 - CustomCNN: A CNN model with three convolutional layers followed by fully connected layers.
 - Dataset Preparation: Custom ImageDataset class for loading and augmenting data.
 - Training: Implements model training, validation, and loss/accuracy monitoring.
 - Evaluation: Tests the model on unseen data and generates a confusion matrix.
 - Visualization: Plots training progress and evaluation metrics.

## Results
After training for 30 epochs, the following metrics are achieved:
![image](https://github.com/user-attachments/assets/82c7bb94-b209-4afb-b518-d809273a8d99)
![image](https://github.com/user-attachments/assets/3e755df5-b4c8-48db-bff2-7cac071006f7)

Test Accuracy: ~83.5%

## Suggestions
 - You can add more epoch e.g 50
 - add more data augmented
 - You can also use ResNet as a comparison 

## Acknowledgments
This project was developed as part of an assignment for the Informatics major at ITENAS Bandung. Special thanks to mentors and peers for their guidance and support.
