# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:31:05 2023

@author: USER
"""

import pandas as pd
import numpy as np
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
plt.style.use('ggplot')
warnings.simplefilter("ignore")

#Get Data
file_loc = r"C:\Users\USER\Desktop\171805024-181805052-ML\indoor_data_HAACS.xlsx"
comdata = pd.read_excel(file_loc)
comdata.head()
#Check Details
comdata.info()

#Total Number of Rows and Columns
rows_col = comdata.shape
print("Total number of records in the dataset : ", rows_col[0])
print("Total number of columns in the dataset : ", rows_col[1])

#Statistical Information
comdata.describe()

#Classification with visualization
sns.set_style("whitegrid")
plt.figure(figsize = (15,8))
plt.xticks(rotation=65,size=10)
sns.countplot(x='X (Numeric)', data=comdata) 
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize = (15,8))
plt.xticks(rotation=65,size=10)
sns.countplot(x='Y (Numeric)', data=comdata) 
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize = (15,8))
plt.xticks(rotation=65,size=10)
sns.countplot(x='Floor (Categoric)', data=comdata) 
plt.show()

comdata['X (Numeric)'].value_counts(normalize=True)*100
comdata['X (Numeric)'].value_counts(normalize=True)*100
comdata['Floor (Categoric)'].value_counts(normalize=True)*100
res = pd.pivot_table(data=comdata, index='Y (Numeric)', columns='X (Numeric)', values='Floor (Categoric)')
res
#Heat map
sns.heatmap(res, cmap='RdYlGn', annot=True)

sns.displot(comdata['X (Numeric)'])
sns.displot(comdata['Y (Numeric)'])
sns.displot(comdata['Floor (Categoric)'])
comdata.corr()
#Seperating Predictors and Response
X = comdata.drop(['X (Numeric)', 'Y (Numeric)', 'Floor (Categoric)'], axis=1)
y=comdata[['X (Numeric)', 'Y (Numeric)' , 'Floor (Categoric)']]
X.head()

#Data Exploration
# Examination of the data 
display(comdata.head()) # starting five index
display(comdata.describe().T) # statistical datas
print(comdata.shape) # length of columns and rows

# Convert 'Floor (Categoric)' to numeric representation
comdata['Floor (Numeric)'] = pd.factorize(comdata['Floor (Categoric)'])[0]

# Select the variables for correlation analysis
variables = ['X (Numeric)', 'Y (Numeric)', 'Floor (Numeric)']

# Calculate the correlation matrix
correlation_matrix = comdata[variables].corr()

# Plot the correlation matrix, relationship of three variables
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#Data Preprocessing
# Check for missing values in the 'X (Numeric)' column
missing_values_x = comdata['X (Numeric)'].isna()

# Check for missing values in the 'Y (Numeric)' column
missing_values_y = comdata['Y (Numeric)'].isna()

# Check for missing values in the 'Floor (Categoric)' column
missing_values_floor = comdata['Floor (Categoric)'].isna()

# Count the number of missing values in each column
num_missing_x = missing_values_x.sum()
num_missing_y = missing_values_y.sum()
num_missing_floor = missing_values_floor.sum()

print(f"Number of missing values in 'X (Numeric)': {num_missing_x}")
print(f"Number of missing values in 'Y (Numeric)': {num_missing_y}")
print(f"Number of missing values in 'Floor (Numeric)': {num_missing_floor}")

from sklearn.preprocessing import StandardScaler
# Handling missing values
comdata.loc[:, 'X (Numeric)'].fillna(comdata['X (Numeric)'].mean(), inplace=True)
comdata.loc[:, 'Y (Numeric)'].fillna(comdata['Y (Numeric)'].mean(), inplace=True)
comdata.loc[:, 'Floor (Categoric)'].fillna('Unknown', inplace=True)

# Scaling numerical features
scaler = StandardScaler()
comdata['X (Numeric)'] = scaler.fit_transform(comdata['X (Numeric)'].values.reshape(-1, 1))
comdata['Y (Numeric)'] = scaler.fit_transform(comdata['Y (Numeric)'].values.reshape(-1, 1))

# Encoding categorical feature
comdata = pd.get_dummies(comdata, columns=['Floor (Numeric)'])

# Checking the preprocessed data
display(comdata.head())

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=37)

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying PCA with n_components = 2
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Functions to visualise Training & Test Set Results
def visualization_train(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('X (Numeric)', 'Y (Numeric)')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('X (Numeric)', 'Y (Numeric)'))(i), label = j)
    plt.title("%s Training Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
def visualization_test(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('X (Numeric)', 'Y (Numeric)')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('X (Numeric)', 'Y (Numeric)'))(i), label = j)
    plt.title("%s Test Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    
#Data Usage
# Splitting the dataset into features (X) and target variable (y)
X = comdata['X (Numeric)']  
y = comdata['Floor (Categoric)']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the X_train array
X_train = np.array(X_train).reshape(-1, 1)

# Applying a machine learning algorithm (e.g., Random Forest Classifier)
model = RandomForestClassifier()  # Replace with the desired classifier

# Training the model on the training set
model.fit(X_train, y_train)

# Making predictions on the testing set
X_test = np.array(X_test).reshape(-1, 1)
y_pred = model.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Printing the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_matrix_result)

#Predicting the Test Set Results
#y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))
# Select the features and target variable
X = comdata[[' F_dB_min', ' F_dB_max', ' F_dB_mean']]
y = comdata['Floor (Categoric)']
# Perform label encoding on the categorical feature
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=37)

classifier.fit(X_train,y_train)

print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.neighbors import KNeighborsClassifier as KNN

classifier = KNN()
classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.naive_bayes import GaussianNB as NB

classifier = NB()
classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.tree import DecisionTreeClassifier as DT

classifier = DT(criterion='entropy',random_state=37)
classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 37)
classifier.fit(X_train, y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
#Predicting the Test Set Results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# Regression Analysis between sales and store
sns.regplot('X (Numeric)', 'Y (Numeric)', comdata)

# Regression Analysis between sales and store
sns.regplot('X (Numeric)', 'Floor (Categoric)', comdata)
# data visualisation for 
sns.violinplot(x="X (Numeric)", y="Y (Numeric)", data=comdata)
# data visualisation for 
sns.violinplot(x="X (Numeric)", y="Floor (Categoric)", data=comdata)

## splitting data into training and test data at 80% and 20% respectively
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
xm_train, xm_test, ym_train, ym_test = train_test_split(X, y, train_size = 0.8, random_state = 100)
#Building model with multiple linear regression
import sklearn.linear_model as lm
lin = lm.LinearRegression()
lin.fit(xm_train, ym_train)
y_pred = lin.predict(xm_test)
print("Mean Square Error: ", mean_squared_error(ym_test, y_pred))
print("Variance or r-squared: ", explained_variance_score(ym_test, y_pred))

#Building model with Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

rfr = RandomForestRegressor(n_estimators=13, random_state=0)
rfr.fit(xm_train, ym_train)
y_predicted = rfr.predict(xm_test)
print("Mean Square Error: ", mean_squared_error(ym_test, y_predicted))
print("Variance or r-squared: ", explained_variance_score(ym_test, y_predicted))

#Building model with Decision Tree Regressor
from sklearn import tree
from sklearn.metrics import mean_squared_error, explained_variance_score

dtr = tree.DecisionTreeRegressor()
dtr.fit(xm_train, ym_train)
ntree = dtr.predict(xm_test)
print("Mean Square Error: ", mean_squared_error(ym_test, ntree))
print("Variance or r-squared: ", explained_variance_score(ym_test, ntree))

#Model Evaluation and Deployment
# Linear Regression 
Dep_pred = lin.predict(X)
# Evaluation
print("mean square error: ", mean_squared_error(y, Dep_pred))
print("variance or r-squared: ", explained_variance_score(y, Dep_pred))

# Random Forest Regressor
Dep = rfr.predict(X)
print("mean square error: ", mean_squared_error(y, Dep))
print("variance or r-squared: ", explained_variance_score(y, Dep))

#Further evaluation using cross validation
# Using cross validation to test the best model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rfr, X, y,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)

#Time and Performance Improvement
# Tuning our model
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X, y)
print(grid_search)
GridSearchCV(cv=5, estimator=RandomForestRegressor(),
             param_grid=[{'max_features': [2, 4, 6, 8],
                          'n_estimators': [3, 10, 30]},
                         {'bootstrap': [False], 'max_features': [2, 3, 4],
                          'n_estimators': [3, 10]}],
             return_train_score=True, scoring='neg_mean_squared_error')

# obtaining the best parameters
grid_search.best_params_

# obtaining the best estimators
grid_search.best_estimator_

# printing all MSE for each parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)
     
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=3)  # Select top k features based on ANOVA F-value
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_

# Cross-Validation
cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=5)

# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Training and Evaluation
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print Results
print("Cross-Validation Scores:", cv_scores)
print("Best Model:", best_model)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.metrics import r2_score

# Calculate R-squared values for X and Y regression models
r2_x_regression = r2_score(y_test, y_pred)
r2_y_regression = r2_score(y_test, y_pred)

print("r2_x_regression    r2_y_regression    accuracy")
print(r2_x_regression,r2_y_regression,accuracy)
print("-"*55)

# Calculate the performance score
project_performance_score = r2_x_regression * r2_y_regression * accuracy

print("Project Performance Score:", project_performance_score)

#Data Correlation
import plotly.graph_objs as go
# Calculate correlation
corr = comdata.corr()

# Create the correlation matrix heatmap
fig = go.Figure(data=go.Heatmap(
                   z=corr.values,
                   x=corr.columns,
                   y=corr.columns,
                   colorscale='Viridis',
                   colorbar=dict(title='Correlation')))

# Update heatmap layout
fig.update_layout(title='Correlation Matrix Heatmap',
                  xaxis=dict(side='top'))

# Show the figure
fig.show()

#K-Means Clustering
# create a list to store the sum of squared distances for each k
import plotly.express as px
ssd = []
scaler = StandardScaler()
full_data = scaler.fit_transform(comdata)
# fit KMeans clustering with different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(full_data)
    ssd.append(kmeans.inertia_)

# create a dataframe with the k values and corresponding ssd
comdata = pd.DataFrame({'k': range(1, 11), 'ssd': ssd})

# create the line plot using Plotly Express
fig = px.line(comdata, x='k', y='ssd', title='Elbow Method')
fig.update_traces(mode='markers+lines', marker=dict(size=8))
fig.show()
