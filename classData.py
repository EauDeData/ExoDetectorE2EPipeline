#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:12:41 2020

@author: gerard
"""

import numpy as np 
import pickle as pk
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Parser:
    """
    Used to train the model, contains the data (light curves) and the labels (class)
    """
    
    def __init__(self, csvFilename, csvMetadataFilename):
        file1 = pd.read_csv(csvFilename)
        estrelles = set(file1['object_id'])
        corves = {x: [] for x in estrelles}
        array = np.array(file1)
        for line in array[:]:
            corves[line[0]].append((line[1], line[3])) #(#timestamp, #flux)
        self.corves = {x: [y[1] for y in sorted(corves[x], key = lambda x: x[0])]  for x in corves}
        file2 = np.array(pd.read_csv(csvMetadataFilename))[:]
        self.categories = {x[0]: int(x[-1]) for x in file2 }
        self.estrelles = list(estrelles)
        self.tamany = max([len(self.corves[x]) for x in corves])
        self.numClases = len(set(self.categories.values()))
        print(self.numClases)
        self.lut = {x: num for num, x in enumerate(set(self.categories.values()))}
        self.maximum = max([max(x) for x in self.corves.values()])
        del file1, corves, array, file2

    def __len__(self):
        return len(self.categories)
    
    def __getitem__(self, index):
        est = self.estrelles[index]
        x = np.zeros(self.tamany)
        x[:len(self.corves[est])] = np.array(self.corves[est])

        #x = (x - x.min())/(x.max() - x.min()) #0 , 1

        y = np.zeros(self.numClases)
        y[self.lut[self.categories[est]]] = 1

        y = y.reshape((1, -1))
        x = x.reshape((1, 1, -1))

        return x, y

def create_model(obj):
    """
    Model used: combination of conv1D and Dense layers
    """
    max_features = 4000000
    max_len = 352
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, obj.numClases, input_length=max_len))
    model.add(tf.keras.layers.Conv1D(32, 7, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool1D(5))
    model.add(tf.keras.layers.Conv1D(64, 7, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(56, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(obj.numClases, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc']
    )
    
    #model.summary()
    return model

def train_model(obj):
    """
    Function used to train the model
    """
   
    x = []
    y = []
    for i in range(len(obj)):
        x.append(obj[i][0][0][0])
        y.append(obj[i][1][0])
    
    x = np.array(x) - np.min(x) # Important fer-ho amb dades noves
        
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train)
    y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train)
    
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test)
    y_test = tf.keras.preprocessing.sequence.pad_sequences(y_test)
    
    X_validation = tf.keras.preprocessing.sequence.pad_sequences(X_validation)
    y_validation = tf.keras.preprocessing.sequence.pad_sequences(y_validation)

    def scheduler(epoch):
      if epoch < 10:
        return 0.001
      else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))
    
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_acc:.4f}.h5', verbose=1)
    ]
    
    model = create_model()
    
    model.fit(X_train, y_train, epochs=30, batch_size=128, callbacks=[callback], validation_data=(X_validation, y_validation))
    
    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=128)
    print('test loss, test acc:', results)
    
def get_obj():
    """
    Function used to load the pickle containing the data used for training/validation/test
    """
    return pk.load(open('datasetRick.p', 'rb'))




