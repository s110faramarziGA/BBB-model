# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:18:18 2023

@author: sadegh-pc
"""

import pandas as pd
# import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras import layers
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
#%%
# load training set
TS=pd.read_excel('D:\Professional\TheProject\Codes\Structures\DataSheet1_Development of QSAR models to predict blood-brain barrier permeability.xlsx', index_col=0)

act=TS['Activity score']
act=act.to_numpy()
#%% 
# fingerprint array
fpArRaw=pd.read_csv('D:\Professional\TheProject\Codes\\fpArray.csv')
fpAr = fpArRaw.drop(fpArRaw.columns[0],axis=1)
# fpArRaw=fpArRaw[['molecular mass',	'mlogp',	'tpsa',	'HAccept',	'HDon',	'nrotate',	'carboxylic acid',	'carboxylate', 'alcohol',	'primary amine',	'secondary amine',	'tertiary amine',	'amide',	'ether',	'Ip3',	'epoxide']]
# fpAr = fpAr.tail(-1)
fpAr=fpAr.to_numpy()
#%% https://www.youtube.com/watch?v=_c_x8A3mNDk&ab_channel=KindsonTheGenius
# fashiondata=tf.keras.datasets.mnist
# (x_train,y_train), (x_test,y_test)=fashiondata.load_data()

# #%% 
# print(x_test.shape)
# print(x_train.shape)

# x_train, x_test=x_train, x_test
# model = tf.keras.models.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10,activation='softmax')
#     ])

# #%%
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
#     )

# #%%
# model.fit(x_train,y_train, epochs=5)
# #%%
# model.evaluate(x_test,y_test)

#%%
model = tf.keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    # keras.layers.Dense(100, activation='softmax'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50,activation='softmax')
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(fpAr[0:900],act[0:900], epochs=1000)
#%% external validation https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
model.fit(fpAr[0:900], act[0:900], validation_data=(fpAr[901:920],act[901:920]), epochs=100, batch_size=10)
#%% external validation
results = model.evaluate(fpAr[901:920], act[901:920], batch_size=10)
print("test loss, test acc:", results)

#%%
# model = keras.Sequential(
#     [
#         keras.layers.Flatten(),
#         RandomFourierFeatures(
#             output_dim=921, scale=1, kernel_initializer="gaussian"
#         ),
#         layers.Dense(units=10),
#     ]
# )
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     loss=keras.losses.hinge,
#     metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
# )
# #%%
# model.fit(fpAr, act, epochs=1000, batch_size=128, validation_split=0.2)

#%% https://vitalflux.com/svm-classifier-scikit-learn-code-examples/
#  ************ SVM ***********


# IRIS Data Set
 
iris = datasets.load_iris()
# X = iris.data
X=fpAr
# y = iris.target
y=act
 
# Creating training and test split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify = y)
 
# Feature Scaling
 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
 
# Training a SVM classifier using SVC class
svm = SVC(kernel= 'rbf', random_state=1, C=1)
svm.fit(X_train_std, y_train)
 
# Mode performance
 
y_pred = svm.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
#specigficity 
recall_score(y_test, y_pred, pos_label=0)

svm.score(X_train, y_train)
#%% cross validation performance https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(svm, X_train, y_train, cv=40)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#%% knn https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=40)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print(accuracy_score(y_test, y_pred))
