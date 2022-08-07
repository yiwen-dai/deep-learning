import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ********************************************************************************************
# DATA PROCESSING
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values      # independent variables
y = dataset.iloc[:, -1].values        # dependent variable

# encode gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# encode geography
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train[:, :])
X_test = sc.transform(X_test[:, :])

# ********************************************************************************************
# ANN
ann = tf.keras.models.Sequential()

# first hidden layer with 6 nodes and relu activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# output layer (activation = 'softmax' if output non binary)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

# ********************************************************************************************
# PREDICTION
y_pred = (ann.predict(X_test) > 0.5)
# output [[predicted, actual]]
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# example custom prediction
# prediction_input = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
# print(ann.predict(sc.transform(
#     [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
