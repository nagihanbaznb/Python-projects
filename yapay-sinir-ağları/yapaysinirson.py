import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv(r"C:\Users\USER\Downloads\Churn_Modelling.csv")

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values    
            
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])  


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
         

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def build_model():
   model = Sequential()
   model.add(Dense(units=6, activation='relu', input_dim=10))  
   model.add(Dense(units=6, activation='relu'))
   model.add(Dense(units=1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

classifier = build_model()
history= classifier.fit(X_train, y_train, epochs=100)

acc_history = history.history["accuracy"]

loss, accuracy = classifier.evaluate(X_test, y_test)

myTest_data = np.array([[2, 500, 0, 42, 8, 150000, 2, 1, 1, 100000]])
myTest_data = sc.transform(myTest_data)

my_predict = classifier.predict(myTest_data)
my_predict = (my_predict > 0.5)

print("Ornek veri tahmini:")
print(my_predict)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
