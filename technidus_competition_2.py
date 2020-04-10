# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:14:31 2019

@author: TERMASHINI
"""

#Technidus competition 2

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


imputer = Imputer(missing_values ="NaN", strategy="mean", axis=0)


test_dataset = pd.read_csv('test_technidus_clf.csv')


train_dataset = pd.read_csv('train_technidus_clf.csv')



#identifying columns with missing values
train_dataset.isnull().sum()

test_dataset.isnull().sum()

cols_with_missing_values = [col for col in train_dataset.columns if train_dataset[col].isnull().any()]

cols_with_missing_values2 = [col for col in test_dataset.columns if test_dataset[col].isnull().any()]


#Filling missing string values in  train and test dataset

train_string_missing_values = ['City', 'StateProvinceName','CountryRegionName','BirthDate', 'Education', 'Occupation', 'Gender', 'MaritalStatus']

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
features_df = pd.get_dummies(all_Data, columns= ['City', 'StateProvinceName','CountryRegionName','BirthDate', 'Education', 'Occupation', 'Gender', 'MaritalStatus'], dummy_na=True)

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

#columns to drop
to_drop = ['Title', 'FirstName', 'MiddleName', 'LastName', 'Suffix', 'AddressLine1', 'AddressLine2', 'PostalCode', 'PhoneNumber']

train_dataset = train_dataset.drop(train_dataset[to_drop], axis = 1)

test_dataset = test_dataset.drop(test_dataset[to_drop], axis = 1)

#Building our Model.........................................

#Selecting  our target and features
data_target = train_dataset.BikeBuyer

data_features = train_dataset.drop(['BikeBuyer'], axis = 1)

test_data_features = test_dataset.drop(['BikeBuyer'], axis = 1)

#validating our model

train_X, val_X, train_y, val_y = train_test_split(data_features, data_target, test_size = 0.3, random_state = 1)

# Define model
predictive_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)

# Fit model
predictive_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = predictive_model.predict(val_X)

# get predicted prices on test data

predictions = predictive_model.predict(test_data_features)

print(mean_absolute_error(val_y, val_predictions))
print (predictions)

my_submission = pd.DataFrame({'CustomerID':test_dataset.CustomerID,'BikeBuyer':predictions.astype(int)})
print(my_submission)
my_submission.to_csv('submission.csv', index=False)