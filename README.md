# Handwritten Digit Recognizer using SVM

This repository contains code for building a handwritten digit recognizer using Support Vector Machines (SVM). The goal is to correctly identify digits from a dataset of handwritten images.

### Dataset

The dataset used for this project is the MNIST dataset available on Kaggle. It consists of 28x28 pixel grayscale images of handwritten digits (0-9). You can find it in this repository as **digit-recognizer** or download it from https://www.kaggle.com/competitions/digit-recognizer/data

### Environment
The code was developed and tested in Google Colab. You can easily set up a Colab environment by following these steps:

Go to Google Colab.
- Create a new notebook.
- Upload your dataset (train.csv and test.csv) to your Colab session.

### Dependencies
Make sure you have the following Python libraries installed:
- numpy
- pandas
- scikit-learn
You can install them using pip: `pip install numpy pandas scikit-learn`

### Usage
1. Open the provided Jupyter notebook (DigitRecognizer.ipynb) in Colab/Jupyternotebook.
2. Make Sure the dataset folder (digit-recognizer) is in the root folder where your jupyter notebook is found.
3. Load the dataset.
4. Preprocess the data (normalize pixel values, split into train and test sets).
5. Train an SVM model with an RBF kernel.
6. Evaluate the model’s accuracy.
7. Make predictions on the test set.

   
Model Hyperparameters : You can fine-tune the SVM hyperparameters (such as C and gamma) to improve accuracy. 

### Results
The trained SVM model achieved an accuracy of 96% on the test set.
