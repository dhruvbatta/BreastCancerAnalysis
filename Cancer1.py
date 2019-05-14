#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:20:45 2019

@author: dhruv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data
data = pd.read_csv('data .csv')

del data['Unnamed: 32']

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1004, random_state = 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(26, activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(0.1))
# Adding the second hidden layer
classifier.add(Dense(13,  activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(0.1))
classifier.add(Dense(6, activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(0.1))
classifier.add(Dense(1,  activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=150, nb_epoch=150,shuffle=True)



y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.savefig('h11.png')


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))