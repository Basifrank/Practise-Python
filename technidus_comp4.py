# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:36:43 2019

@author: TERMASHINI
"""


#Technidus competition 2

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import datetime
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier




tdate = datetime.date.today()
Current_year = tdate.year

imputer = Imputer(missing_values ="NaN", strategy="mean", axis=0)


test_dataset = pd.read_csv('test_technidus_clf.csv')


train_dataset = pd.read_csv('train_technidus_clf.csv')



#identifying columns with missing values
train_dataset.isnull().sum()

test_dataset.isnull().sum()

cols_with_missing_values = [col for col in train_dataset.columns if train_dataset[col].isnull().any()]

cols_with_missing_values2 = [col for col in test_dataset.columns if test_dataset[col].isnull().any()]


#Filling missing string values in  train and test dataset

train_string_missing_values = ['City', 'StateProvinceName','BirthDate', 'CountryRegionName','Education', 'Occupation', 'Gender', 'MaritalStatus']

train_dataset[train_string_missing_values] = train_dataset[train_string_missing_values].fillna(train_dataset.mode().iloc[0])


test_dataset[train_string_missing_values] = test_dataset[train_string_missing_values].fillna(test_dataset.mode().iloc[0])


#Filling missing numeric values in  train dataset
train_numeric_missing_values = ['HomeOwnerFlag','CustomerID', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'AveMonthSpend']


#fitting and trasforming numeric data

imputer.fit(train_dataset[train_numeric_missing_values])
train_dataset[train_numeric_missing_values] = imputer.fit_transform(train_dataset[train_numeric_missing_values])

imputer.fit(test_dataset[train_numeric_missing_values])
test_dataset[train_numeric_missing_values] = imputer.fit_transform(test_dataset[train_numeric_missing_values])

#combining all data
da_ta = [train_dataset , test_dataset]

all_Data = pd.concat(da_ta, keys=['x', 'y'])

# Create your dummies
features_df = pd.get_dummies(all_Data, columns= ['City', 'StateProvinceName','CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus'], dummy_na=True)

#splitting all data

train_dataset = features_df.loc['x']

test_dataset = features_df.loc['y']

#...........use get dummies to one hot encode

#train_dataset_categorical = pd.get_dummies(data = train_dataset[train_string_missing_values])

#test_dataset_categorical = pd.get_dummies(data = test_dataset[train_string_missing_values])

#combining all data to do the 


#dropping categorical data
#train_dataset = train_dataset.drop(train_dataset[train_string_missing_values], axis = 1)

#test_dataset = test_dataset.drop(test_dataset[train_string_missing_values], axis = 1)


#merging categorical data with other data
#train_dataset = pd.concat([train_dataset, train_dataset_categorical], axis=1)

#test_dataset = pd.concat([test_dataset, test_dataset_categorical], axis=1)

#converting strings date to date time and feature engineering

train_dataset['Birth_Date'] = pd.to_datetime(train_dataset['BirthDate'], infer_datetime_format = True)

test_dataset['Birth_Date'] = pd.to_datetime(test_dataset['BirthDate'], infer_datetime_format = True)



#Getting Age of customers
train_dataset['Age'] = Current_year - train_dataset['Birth_Date'].dt.year 

test_dataset['Age'] = Current_year - test_dataset['Birth_Date'].dt.year 



#doing an exploratory data analysis on our data
print(train_dataset['BikeBuyer'].astype(int).plot.hist(bins = 20))

print(train_dataset['BikeBuyer'].value_counts())


print(train_dataset.dtypes.value_counts())
print(test_dataset.dtypes.value_counts())

print(train_dataset.shape)

print(train_dataset['YearlyIncome'].describe())


print(train_dataset['AveMonthSpend'].describe())

print(train_dataset['TotalChildren'].describe())

# Find correlations with the target and sort
correlations = train_dataset.corr()['BikeBuyer'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

print(correlations)

print(train_dataset['Age'].plot.hist(bins = 20))



#Finding Outliers with boxplot
import seaborn as sns
sns.boxplot(x=train_dataset['BikeBuyer'], y=train_dataset['AveMonthSpend'], data=train_dataset, palette = 'hls')
sns.boxplot(x=train_dataset['AveMonthSpend'])
sns.boxplot(x=train_dataset['Age'])
train_dataset[train_dataset['AveMonthSpend']>140].count()
train_dataset['Age'].hist(bins = 20)
sns.boxplot(x=train_dataset['YearlyIncome'])

#Handling Outliers using log, sqrt and cube root
#with Age_log
train_dataset['Age_Log'] = np.log(train_dataset['Age'])
train_dataset['Age_Log'].hist(bins = 20)

test_dataset['Age_Log'] = np.log(test_dataset['Age'])
test_dataset['Age_Log'].hist(bins = 20)


#with Age_sqrt
train_dataset['Age_sqrt'] = np.sqrt(train_dataset['Age'])
train_dataset['Age_sqrt'].hist(bins = 20)

test_dataset['Age_sqrt'] = np.sqrt(test_dataset['Age'])
test_dataset['Age_sqrt'].hist(bins = 20)

#with Age_cbrt
train_dataset['Age_cbrt'] = np.cbrt(train_dataset['Age'])
train_dataset['Age_cbrt'].hist(bins = 20)

test_dataset['Age_cbrt'] = np.cbrt(test_dataset['Age'])
test_dataset['Age_cbrt'].hist(bins = 20)



#with AveMonthSpend_Log

train_dataset['AveMonthSpend_Log'] = np.log(train_dataset['AveMonthSpend'])
train_dataset['AveMonthSpend_Log'].hist(bins = 20)

test_dataset['AveMonthSpend_Log'] = np.log(test_dataset['AveMonthSpend'])
test_dataset['AveMonthSpend_Log'].hist(bins = 20)

#with AveMonthSpend_sqrt

train_dataset['AveMonthSpend_sqrt']  = np.sqrt(train_dataset['AveMonthSpend'])
train_dataset['AveMonthSpend_sqrt'].hist(bins = 20)

test_dataset['AveMonthSpend_sqrt'] = np.sqrt(test_dataset['AveMonthSpend'])
test_dataset['AveMonthSpend_sqrt'].hist(bins = 20)


#with AveMonthSpend_cbrt

train_dataset['AveMonthSpend_cbrt']  = np.cbrt(train_dataset['AveMonthSpend'])
train_dataset['AveMonthSpend_cbrt'].hist(bins = 20)

test_dataset['AveMonthSpend_cbrt'] = np.cbrt(test_dataset['AveMonthSpend'])
test_dataset['AveMonthSpend_cbrt'].hist(bins = 20)


#columns to drop
to_drop = ['Title', 'BirthDate','Age', 'Age_cbrt', 'Age_sqrt', 'AveMonthSpend_cbrt','AveMonthSpend_sqrt', 'Birth_Date', 'FirstName', 'MiddleName', 'LastName', 'Suffix', 'AddressLine1', 'AddressLine2', 'PostalCode', 'PhoneNumber', 'City_Bountiful', 'City_Cedar City', 'City_Citrus Heights','City_City Of Commerce','City_Clearwater','City_nan','StateProvinceName_Florida','StateProvinceName_Utah','StateProvinceName_nan','CountryRegionName_nan','Education_nan','Occupation_nan','Gender_nan','MaritalStatus_nan' ]

train_dataset = train_dataset.drop(train_dataset[to_drop], axis = 1)

test_dataset = test_dataset.drop(test_dataset[to_drop], axis = 1)


#Selecting  our target and features
data_target = train_dataset.BikeBuyer

data_features = train_dataset.drop(['BikeBuyer'], axis = 1)

test_data_features = test_dataset.drop(['BikeBuyer'], axis = 1)

#validating our model

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.4, random_state=4)

#Recursive Feature Elimination############################################

#Defining our model and create a base classifier to evaluate a subset of attibutes
#model = LogisticRegression()

#create the RFE model and select 3 attributes
#rfe = RFE(model, 3)
#rfe = rfe.fit(data_features, data_target)

#Summarise the selection of the attribute
#print(rfe.support_)
#print(rfe.ranking_)

#Feauture Importance to select feature................................
#model2 = ExtraTreesClassifier()

#fit an extra trees model to the data
#model2.fit(data_features, data_target)

#Summarise the selection of the attribute
#print(model2.feature_importances_)
#ridge model
#from sklearn.linear_model import Ridge
#model3 = Ridge(alpha=0.05, normalize=True)

## training the model


#Defining our model and create a base classifier to evaluate a subset of attibutes
model = LogisticRegression()
model2 = KNeighborsClassifier(n_neighbors = 5)
model3 = DecisionTreeClassifier(random_state=4)
# Fit and train model
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
# get predicted prices on validation data
val_predictions = model.predict(X_test)
val_predictions2 = model2.predict(X_test)
val_predictions3 = model3.predict(X_test)

print (metrics.accuracy_score(y_test, val_predictions))

print (metrics.accuracy_score(y_test, val_predictions2))

print (metrics.accuracy_score(y_test, val_predictions3))

#finding the best value for k
k_range = range(50,500)
scores = []
for k in k_range:
    model4 = DecisionTreeClassifier(max_leaf_nodes=k, random_state=4)
    model4.fit(X_train, y_train)
    val_Pred = model4.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, val_Pred))

%matplotlib inline

plt.plot(k_range, scores)
plt.xlabel('value of K for model4')
plt.ylabel('testing Accuracy')


'City_Bountiful', 'City_Cedar City', 'City_Citrus Heights'                           
,'City_City Of Commerce'
,'City_Clearwater'                               
,'City_nan'                                      
,'StateProvinceName_Florida'                     
,'StateProvinceName_Utah'                        
,'StateProvinceName_nan'                         
,'CountryRegionName_nan'                         
,'Education_nan'                                 
,'Occupation_nan'                                
,'Gender_nan'                                    
,'MaritalStatus_nan'                             










# get predicted prices on test data

predictions = model3.predict(test_data_features)

print(mean_absolute_error(y_test, val_predictions))
print (predictions)

my_submission = pd.DataFrame({'CustomerID':test_dataset.CustomerID,'BikeBuyer':predictions.astype(int)}).set_index('CustomerID')
print(my_submission)
my_submission.to_csv('4th_submission.csv', index=False)
