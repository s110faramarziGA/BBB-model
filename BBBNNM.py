# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:18:18 2023

@author: sadegh-pc
"""

# Importing necessary libraries
import pandas as pd
# import os  # Uncomment if os functionalities are needed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn import datasets
import os

#%% Load training set
# Set working directory to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Load dataset from an Excel file. This dataset includes a column 'Activity score'.
TS = pd.read_excel('DataSheet1_Development of QSAR models to predict blood-brain barrier permeability.xlsx', index_col=0)

# Extract the 'Activity score' column and convert it to a NumPy array
act = TS['Activity score']
act = act.to_numpy()

#%% Load fingerprint array
# Read feature array data from a CSV file and drop the first column (likely an ID or index column)
fpArRaw = pd.read_csv('D:\\Codes\\fpArray.csv')
fpAr = fpArRaw.drop(fpArRaw.columns[0], axis=1)  # Remove the first column
fpAr = fpAr.to_numpy()  # Convert the DataFrame to a NumPy array

#%% Build and train a neural network model using TensorFlow
model = tf.keras.models.Sequential([
    keras.layers.Flatten(),  # Flatten input data (no specific shape required)
    keras.layers.Dense(10, activation='relu'),  # Add a Dense layer with 10 units and ReLU activation
    keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
    keras.layers.Dense(50, activation='softmax')  # Output layer with 50 units and softmax activation
])

# Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metrics
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using the first 900 samples for 1000 epochs
model.fit(fpAr[0:900], act[0:900], epochs=1000)

#%% External validation
# Train the model and validate it on samples 901 to 920
model.fit(fpAr[0:900], act[0:900], validation_data=(fpAr[901:920], act[901:920]), epochs=100, batch_size=10)

# Evaluate the model on validation samples and print the test loss and accuracy
results = model.evaluate(fpAr[901:920], act[901:920], batch_size=10)
print("test loss, test acc:", results)

#%% Train and evaluate an SVM classifier
# Load the IRIS dataset for reference (not used here)
iris = datasets.load_iris()

# Use the fingerprint array (fpAr) as features and 'Activity score' (act) as labels
X = fpAr
y = act

# Split the dataset into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

# Standardize the features for SVM
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train an SVM classifier with radial basis function (RBF) kernel
svm = SVC(kernel='rbf', random_state=1, C=1)
svm.fit(X_train_std, y_train)

# Evaluate SVM performance
y_pred = svm.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))  # Print accuracy
print('Specificity:', recall_score(y_test, y_pred, pos_label=0))  # Calculate specificity

# Evaluate SVM on training data
print('SVM training score:', svm.score(X_train, y_train))

#%% Perform cross-validation
# Perform 40-fold cross-validation on the training set
scores = cross_val_score(svm, X_train, y_train, cv=40)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% Train and evaluate a k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

# Train a k-NN classifier with 40 neighbors
neigh = KNeighborsClassifier(n_neighbors=40)
neigh.fit(X_train, y_train)

# Evaluate k-NN performance on the test set
y_pred = neigh.predict(X_test)
print('k-NN Accuracy:', accuracy_score(y_test, y_pred))  # Print accuracy
