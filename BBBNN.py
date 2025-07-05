# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:18:18 2023

@author: sadegh-pc
"""

import pandas as pd
# import os
import tensorflow as tf
from tensorflow import keras

#%%
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
    keras.layers.Dense(100,activation='softmax')
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(fpAr,act, epochs=1000)