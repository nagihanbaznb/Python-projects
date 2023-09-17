# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:33:11 2023

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
# We begin by importing all the necessary libraries needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#regression packages
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

#lasso regression
from sklearn import linear_model

#f_regression (feature selection)
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

# recursive feature selection (feature selection)
from sklearn.feature_selection import RFE

#ignore warning
import warnings
warnings.filterwarnings("ignore")
file_loc = r"C:\Users\USER\Desktop\171805024-181805052-ML\credit_customers (1).xlsx"
credit_customers_data = pd.read_excel(file_loc)
credit_customers_data.head()
rows_col = credit_customers_data.shape
print("Total number of records in the dataset : ", rows_col[0])
print("Total number of columns in the dataset : ", rows_col[1])

credit_customers_data.info()

n = 1
for (columnName, columnData) in credit_customers_data.iteritems():
    if columnData.dtype == 'O':
        print('S.no   : ', n)
        print('Name   : ', columnName)
        print('Unique : ', columnData.unique())
        print('No     : ',len(columnData.unique()))
        print()
        n+=1
    else:
        pass
    
credit_customers_data[['sex', 'marriage']] = credit_customers_data.personal_status.str.split(" ", expand = True)
credit_customers_data.drop(['personal_status'], axis=1, inplace = True)

credit_customers_data['checking_status'].replace(['no checking', '<0', '0<=X<200', '>=200'], [0,1,2,3], inplace = True)
credit_customers_data['credit_history'].replace(['critical/other existing credit', 'delayed previously' , 'existing paid', 'no credits/all paid', 'all paid'], [0,1,2,2,2], inplace = True)
credit_customers_data['purpose'].replace(['business', 'new car','used car', 'education', 'retraining', 'other','domestic appliance','radio/tv','furniture/equipment','repairs'], [5,5,4,4,3,3,3,2,2,1], inplace = True)
credit_customers_data['savings_status'].replace(['no known savings', '<100','100<=X<500','500<=X<1000','>=1000'], [0,1,2,3,4], inplace = True)
credit_customers_data['employment'].replace(['unemployed', '<1','1<=X<4','4<=X<7','>=7'], [0,1,2,3,4], inplace = True)
credit_customers_data['other_parties'].replace(['none', 'co applicant', 'guarantor'], [0,1,2], inplace = True)
credit_customers_data['property_magnitude'].replace(['no known property', 'life insurance', 'car', 'real estate'], [0,1,2,3], inplace = True)
credit_customers_data['other_payment_plans'].replace(['none', 'stores', 'bank'], [0,1,1], inplace = True)
credit_customers_data['housing'].replace(['for free', 'rent', 'own'], [0,1,2], inplace = True)
credit_customers_data['job'].replace(['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'], [0,1,2,3], inplace = True)
credit_customers_data['own_telephone'].replace(['yes', 'none'], [1,0], inplace = True)
credit_customers_data['foreign_worker'].replace(['yes', 'no'], [1,0], inplace = True)
credit_customers_data['class'].replace(['good', 'bad'], [1,0], inplace = True)
credit_customers_data['sex'].replace(['male', 'female'], [1,0], inplace = True)
credit_customers_data['marriage'].replace(['single', 'div/sep','div/dep/mar','mar/wid'], [0,0,1,1], inplace = True)

data2 = pd.DataFrame(credit_customers_data.groupby(['sex'])['marriage'].sum()).reset_index()
plt.figure(figsize=(10,5))
ax = sns.lineplot(x = 'sex', y='marriage', data = data2)

credit_customers_data.groupby('sex').sum().sort_values(by='marriage', ascending=False)

print(credit_customers_data['class'].value_counts())
from sklearn.preprocessing import StandardScaler
X = credit_customers_data.drop(['class'], axis=1)
y = credit_customers_data['class']
std_scaler = StandardScaler()
Xa = std_scaler.fit_transform(X)
#
sns.set_style("whitegrid")
plt.figure(figsize = (15,8))
plt.xticks(rotation=65,size=10)
sns.countplot(x='class', data=credit_customers_data) 
plt.show()

X = pd.DataFrame(Xa, columns = X.columns)
credit_customers_data.columns

#Model Building and Selection
# splitting data into training and test data at 80% and 20% respectively
from sklearn.model_selection import train_test_split
xm_train, xm_test, ym_train, ym_test = train_test_split(X, y, train_size = 0.8, random_state = 100)
#Building model with multiple linear regression
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

#Tuning our model before deployment
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

# obtaining the best parameters
grid_search.best_params_
# obtaining the best estimators
grid_search.best_estimator_
# printing all MSE for each parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

xm_train = sc.fit_transform(xm_train)
xm_test = sc.transform(xm_test)

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(ym_train,classifier.predict(xm_train))))
        print('Classification Report:\n{}\n'.format(classification_report(ym_train,classifier.predict(xm_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(ym_train,classifier.predict(xm_train))))
        res = cross_val_score(classifier, xm_train, ym_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(ym_test,classifier.predict(xm_test))))
        print('Classification Report:\n{}\n'.format(classification_report(ym_test,classifier.predict(xm_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(ym_test,classifier.predict(xm_test))))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(xm_train,ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=37)

classifier.fit(xm_train,ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)

from sklearn.neighbors import KNeighborsClassifier as KNN

classifier = KNN()
classifier.fit(xm_train,ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)

from sklearn.naive_bayes import GaussianNB as NB

classifier = NB()
classifier.fit(xm_train,ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)

from sklearn.tree import DecisionTreeClassifier as DT

classifier = DT(criterion='entropy',random_state=37)
classifier.fit(xm_train,ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 37)
classifier.fit(xm_train, ym_train)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=True)
print_score(classifier,xm_train,ym_train,xm_test,ym_test,train=False)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.heatmap(credit_customers_data.corr())

# df.drop(['credit_amount','duration'],axis=1,inplace=True)

plt.show()

pd.plotting.scatter_matrix(credit_customers_data, alpha=0.3,figsize=(15,8),diagonal='kde' )
plt.tight_layout()
