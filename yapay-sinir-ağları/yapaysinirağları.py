# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:28:39 2023

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv(r"C:\Users\USER\Downloads\Churn_Modelling.csv")

# Preprocessing
# Remove the one-hot encoding step
X = data.iloc[:, 3:13].values  # Features: columns 3 to 12
y = data.iloc[:, 13].values   # Target: column 13 (Exited)

# Encode categorical variables
label_encoder_X_country = LabelEncoder()
X[:, 1] = label_encoder_X_country.fit_transform(X[:, 1])
label_encoder_X_gender = LabelEncoder()
X[:, 2] = label_encoder_X_gender.fit_transform(X[:, 2])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=10))  # Adjust the input_dim based on the number of features
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=10, epochs=100)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100}%")
