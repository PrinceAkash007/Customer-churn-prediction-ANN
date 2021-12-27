# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.getcwd()

os.chdir('D:\DataScience\Datasets')
print(os.getcwd())

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()

X = dataset.iloc[:, 3:13].values
X

y = dataset.iloc[:, 13].values
y

# Dummy Vars & Encoders
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset2 = ['Pizza','Burger','Bread','Bread','Bread','Burger','Pizza','Burger']

values = array(dataset2)
print(values)


label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)

onehot = OneHotEncoder(sparse=False)
onehot_encoded = onehot.fit_transform(integer_encoded)
print(onehot_encoded)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

print(X)


X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
X

print(X.shape)

tmpDF = pd.DataFrame(X)
tmpDF.head()


from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Geography",OneHotEncoder(),[1])], remainder= 'passthrough')

X


print(X.shape)

X = ct.fit_transform(X)

print(X.shape)

tmpDF = pd.DataFrame(X)
tmpDF.head()

X

print(X.shape)

X = X[:, 1:]

print(X.shape)

tmpDF2 = pd.DataFrame(X)
tmpDF2.head()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

tmpDF = pd.DataFrame(X_train)
tmpDF

X_train = sc.fit_transform(X_train)
# takes a col
# calc the mean
# calc the sd
# sub the mean
# div by sd
tmpDF = pd.DataFrame(X_train)
tmpDF

tmpDF.shape

X_test = sc.transform(X_test)

#  Now let's make the ANN!

# Importing the Keras libraries and packages
#!conda install keras
#import tensorflow
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras import *

# Initialising the ANN
classifier = Sequential()

#classifier.summary()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#classifier.summary()
# Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

classifier.summary()

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = SGD(), loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

classifier.summary()

#history
#  Making predictions and evaluating the model

# Predicting the Test set results
#y_pred = classifier.predict(X_test)
X_test

X_test = np.array(X_test).astype(np.float32)
X_test

y_test = np.array(y_test).astype(np.float32)
y_test 

y_pred = classifier.predict(X_test)
y_pred

y_pred = (y_pred > 0.5)
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])

# change the epochs to 5, 10 from 2
# got 79% acc with 2 & 5 & 20 epochs with SGD
# got 83% acc with 20 epochs with adam



# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)
X_test.shape

X_test

X_test = np.array(X_test).astype(np.float32)
X_test

y_test = np.array(y_test).astype(np.float32)
#y_pred = classifier.predict(X_test)
y_pred = classifier.predict(np.array(X_test).astype(np.float32))
y_pred

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])


# Neurons
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])