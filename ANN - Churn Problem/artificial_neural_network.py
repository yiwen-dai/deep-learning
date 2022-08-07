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

print('Training started...')
ann.fit(X_train, y_train, batch_size=32, epochs=100)
print('Training finished')

# ********************************************************************************************
# PREDICTION
pred_type = input(
    'Would you like to use the neural network to predict the test data (T) or predict custom input data (C)? \n')
pred_type_determined = False

while not pred_type_determined:
    if pred_type == 't' or pred_type == 'T':
        pred_type_determined = True

        y_pred = (ann.predict(X_test) > 0.5)
        print('Predicted output vs test output: \n')
        print(np.concatenate((y_pred.reshape(len(y_pred), 1),
              y_test.reshape(len(y_test), 1)), 1))

        # confusion matrix and accuracy score
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion matrix: \n')
        print(cm)
        accuracy_score(y_test, y_pred)

    elif pred_type == 'c' or pred_type == 'C':
        pred_type_determined = True

        # data
        data = []
        accepted = False

        geo = input('Geographical location (country): ')
        while (not accepted):
            if geo.upper() == 'FRANCE':
                data.append(1)
                data.append(0)
                data.append(0)
                accepted = True
            elif geo.upper() == 'GERMANY':
                data.append(0)
                data.append(0)
                data.append(1)
                accepted = True
            elif geo.upper() == 'SPAIN':
                data.append(0)
                data.append(0)
                data.append(0)
                accepted = True
            else:
                geo = input(
                    'Country not recognized. Please input one of the following countries: [France, Spain, Germany]: ')

        accepted = False
        cs = input('Credit score: ')
        while (not accepted):
            if int(cs) < 300 or int(cs) > 850:
                cs = input(
                    'Invalid credit score value. Please input a valid score between 300 and 850: ')
            else:
                data.append(int(cs))
                accepted = True

        accepted = False
        gender = input('Gender: ')
        while (not accepted):
            if gender.upper() == 'MALE' or gender.upper() == 'M':
                accepted = True
                data.append(1)
            elif gender.upper() == 'FEMALE' or gender.upper() == 'F':
                accepted = True
                data.append(0)
            else:
                gender = input(
                    'Unrecognized input. Please input either male or female: ')

        accepted = False
        age = input('Age: ')
        while not accepted:
            if int(age) > 0:
                accepted = True
                data.append(int(age))
            else:
                age = input('Invalid age. Please input age greater than 0: ')

        accepted = False
        tenure = input('Tenure: ')
        while not accepted:
            if int(tenure) > 0:
                accepted = True
                data.append(int(tenure))
            else:
                tenure = input(
                    'Invalid tenure. Please input tenure (years) greater than 0: ')

        accepted = False
        balance = input('Account balance($): ')
        if balance[0] == '$':
            balance = balance[1:]
        while not accepted:
            if int(balance) > 0:
                accepted = True
                data.append(int(balance))
            else:
                age = input(
                    'Invalid account balance. Please input account balance greater than 0: ')

        accepted = False
        num_prods = input('Number of bank products: ')
        while not accepted:
            if int(num_prods) > 0:
                accepted = True
                data.append(int(num_prods))
            else:
                age = input(
                    'Invalid account balance. Please input account balance greater than 0: ')

        accepted = False
        cred_card = input('Does the customer have a credit card (Y/N): ')
        while not accepted:
            if cred_card.upper() == 'Y' or cred_card.upper() == 'YES':
                accepted = True
                data.append(1)
            elif cred_card.upper() == 'N' or cred_card.upper() == 'NO':
                accepted = True
                data.append(0)
            else:
                age = input(
                    'Unrecognized input. Please input \'Y\' for yes and \'N\' for no: ')

        accepted = False
        active = input('Is this customer an active member (Y/N): ')
        while not accepted:
            if active.upper() == 'Y' or active.upper() == 'YES':
                accepted = True
                data.append(1)
            elif active.upper() == 'N' or active.upper() == 'NO':
                accepted = True
                data.append(0)
            else:
                age = input(
                    'Unrecognized input. Please input \'Y\' for yes and \'N\' for no: ')

        accepted = False
        est_salary = input('Estimated yearly salary($): ')
        if est_salary[0] == '$':
            est_salary = est_salary[1:]
        while not accepted:
            if int(est_salary) > 0:
                accepted = True
                data.append(int(est_salary))
            else:
                age = input(
                    'Invalid salary. Please input account balance greater than 0: ')

        print('Is the customer predicted to leave the bank: ')
        print(ann.predict(sc.transform([data])) > 0.5)

    else:
        pred_type = input(
            'Input unrecognized. Please enter \'T\' to run the test data or enter \'C\' to input custom data: \n')


# example custom prediction
# prediction_input = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
# print(ann.predict(sc.transform(
#     [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
