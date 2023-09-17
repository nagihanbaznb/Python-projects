# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:57:21 2023

@author: USER
"""

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

file_loc = r"C:\Users\USER\Desktop\171805024-181805052-ML\Walmart (2).xlsx"
df = pd.read_excel(file_loc)

df.head()
# Finding information about the dataset
df.info()
# converting date object to datetime
df['Date'] = pd.to_datetime(df.Date)
df.head()
# Reframing the columns by breaking the date into weeks, month and year for analysis

df['weekday'] = df.Date.dt.weekday
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year

df.drop(['Date'], axis=1, inplace=True)#,'month'

target = 'Weekly_Sales'
features = [i for i in df.columns if i not in [target]]
original_df = df.copy(deep=True)

df.head()
# Checking for data types
df.info()
# checking for unique values
df.nunique()
# Checking for missing values in each column
df.isnull().sum()
# Finding the total sales
df.Weekly_Sales.sum()

# The stores with the highest weekly sales
df.groupby(['Store'])['Weekly_Sales'].sum().sort_values(ascending=False).head(5)
# The stores with the highest weekly sales visualization
df.groupby(['Store'])['Weekly_Sales'].sum().sort_values(ascending=False).head(5).plot(kind='bar')

# Summary Statistics
df.describe().T

# Histogram for all columns
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()

#Let us analyze the distribution of the target variable

plt.figure(figsize=[8,4])
sns.distplot(df[target], color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution')
plt.show()

# Correlation Analysis
df.corr()

# Finding the the predictor with the highest relatioship with sales
corr = pd.DataFrame(df.corr()['Weekly_Sales'].drop('Weekly_Sales'))
corr.sort_values(['Weekly_Sales'], ascending = False)
# correlation heatmap
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
# Regression Analysis between sales and store
sns.regplot('Store', 'Weekly_Sales', df)
# Regression Analysis between sales and Holiday_Flag
sns.regplot('Holiday_Flag', 'Weekly_Sales', df)
# data visualisation for 
sns.violinplot(x="Holiday_Flag", y="Weekly_Sales", data=df)
# Categorical to Dummy Variables
df =  pd.get_dummies(df, columns=["Store", "Holiday_Flag", "weekday", "month", "year"],
                         prefix=["Store", "Holiday_Flag", "weekday", "month", "year"],
                         drop_first=True)
df.head(2)

# Convert columns to numeric types
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Fuel_Price'] = pd.to_numeric(df['Fuel_Price'], errors='coerce')
df['Unemployment'] = pd.to_numeric(df['Unemployment'], errors='coerce')

# Replace missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Replace infinite values with a finite value (e.g., 0)
df.replace([np.inf, -np.inf], 0, inplace=True)

### c. checking for p-value (TESTING FOR STATISTICAL SIGNIFICANCE OF INDEPENDENT VARIABLES for variable selection)
import scipy.stats as stats
df_corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame() # Matrix of p-values
for x in df.columns:   # assuming df as your dataframe name
   for y in df.columns:
      corr = stats.pearsonr(df[x], df[y])
      df_corr.loc[x,y] = corr[0]
      df_p.loc[x,y] = corr[1]

df_p['Weekly_Sales']

#Model Building and Selection
# Separating target variable and predictors
y = df ['Weekly_Sales']
x = df.drop(['Weekly_Sales'], axis =1)
# Normalization data to bring all values to common scale
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(x).transform(x)
X[0:1]
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
#Model Evaluation and Deployment
# Linear Regression 
Dep_pred = lin.predict(X)
# Evaluation
print("mean square error: ", mean_squared_error(y, Dep_pred))
print("variance or r-squared: ", explained_variance_score(y, Dep_pred))
# Actual vs Predicted
c = [i for i in range(1, 6436, 1)]
fig = plt.figure()
plt.plot(c, y, color = "blue", linewidth = 2.5, linestyle = "-" )
plt.plot(c, Dep_pred, color = "red", linewidth = 2.5, linestyle = "-" )
fig.suptitle('Actual and Predicted', fontsize = 20)
plt.xlabel('Index', fontsize = 18)
plt.ylabel('medv', fontsize = 16)
# Random Forest Regressor
Dep = rfr.predict(X)
print("mean square error: ", mean_squared_error(y, Dep))
print("variance or r-squared: ", explained_variance_score(y, Dep))
# Actual vs Predicted
c = [i for i in range(1, 6436, 1)]
fig = plt.figure()
plt.plot(c, y, color = "blue", linewidth = 2.5, linestyle = "-" )
plt.plot(c, Dep, color = "red", linewidth = 2.5, linestyle = "-" )
fig.suptitle('Actual and Predicted', fontsize = 20)
plt.xlabel('Index', fontsize = 18)
plt.ylabel('medv', fontsize = 16)
# Top five most influential variables
feature_importances = pd.DataFrame(dtr.feature_importances_, index=x.columns, columns=['importance']).sort_values('importance', ascending=False).head()

feature_importances
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