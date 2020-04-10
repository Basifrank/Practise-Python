# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 04:04:57 2019

@author: TERMASHINI
"""


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
import matplotlib as mpl
%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno



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

#converting birthdate to date type

train_dataset['Birth_Date'] = pd.to_datetime(train_dataset['BirthDate'], infer_datetime_format = True)

test_dataset['Birth_Date'] = pd.to_datetime(test_dataset['BirthDate'], infer_datetime_format = True)



#Getting Age of customers
train_dataset['Age'] = Current_year - train_dataset['Birth_Date'].dt.year 

test_dataset['Age'] = Current_year - test_dataset['Birth_Date'].dt.year 


#Identifyiny numerical data
numeric_features = train_dataset.select_dtypes(include=[np.number])

numeric_features.columns

#Identifying categorical data

categorical_features = train_dataset.select_dtypes(include=[np.object])

categorical_features.columns

#Exploratory data analysis
train_dataset.describe()
train_dataset.head()

train_dataset.skew()    #understanding Skewness
test_dataset.skew()


train_dataset.kurt()    #understanding kurtness
test_dataset.kurt()






#combining all data
da_ta = [train_dataset , test_dataset]

all_Data = pd.concat(da_ta, keys=['x', 'y'])

# Create your dummies
features_df = pd.get_dummies(all_Data, columns= ['City', 'StateProvinceName','CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus'], dummy_na=True)

#splitting all data

train_dataset = features_df.loc['x']

test_dataset = features_df.loc['y']



#doing an exploratory data analysis on our data
print(train_dataset['BikeBuyer'].astype(int).plot.hist(bins = 20))

print(train_dataset['BikeBuyer'].value_counts())


print(train_dataset.dtypes.value_counts())
print(test_dataset.dtypes.value_counts())

print(train_dataset.shape)

print(train_dataset['YearlyIncome'].describe())


print(train_dataset['AveMonthSpend'].describe())

print(train_dataset['TotalChildren'].describe())


print(train_dataset['Age'].plot.hist(bins = 20))

#columns to drop
to_drop = ['Title', 'Birth_Date', 'BirthDate','FirstName', 'MiddleName', 'LastName', 'Suffix', 'AddressLine1', 'AddressLine2', 'PostalCode', 'PhoneNumber', 'City_Bountiful', 'City_Cedar City', 'City_Citrus Heights','City_City Of Commerce','City_Clearwater','City_nan','StateProvinceName_Florida','StateProvinceName_Utah','StateProvinceName_nan','CountryRegionName_nan','Education_nan','Occupation_nan','Gender_nan','MaritalStatus_nan' ]

train_dataset = train_dataset.drop(train_dataset[to_drop], axis = 1)

test_dataset = test_dataset.drop(test_dataset[to_drop], axis = 1)


#Finding Outliers with boxplot
import seaborn as sns
sns.boxplot(x=train_dataset['BikeBuyer'], y=train_dataset['AveMonthSpend'], data=train_dataset, palette = 'hls')
sns.boxplot(x=train_dataset['AveMonthSpend'])
sns.boxplot(x=train_dataset['Age'])
train_dataset[train_dataset['AveMonthSpend']>140].count()
train_dataset['Age'].hist(bins = 20)
sns.boxplot(x=train_dataset['YearlyIncome'])

#replacing outliers with mean in AveMonthSpend
#col_morethan140 = train_dataset[train_dataset['AveMonthSpend']>140].columns

#imputer.fit(train_dataset[col_morethan140])
#train_dataset[col_morethan140] = imputer.fit_transform(train_dataset[col_morethan140])

#imputer.fit(test_dataset[col_morethan140])
#test_dataset[col_morethan140] = imputer.fit_transform(test_dataset[col_morethan140])
# Find correlations with the target and sort
correlations = train_dataset.corr()['BikeBuyer'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

print(correlations)

#Selecting  our target and features
data_target = train_dataset.BikeBuyer

data_features = train_dataset.drop(['BikeBuyer'], axis = 1)

test_data_features = test_dataset.drop(['BikeBuyer'], axis = 1)

features_select = ['NumberChildrenAtHome','MaritalStatus_M','Gender_M','CountryRegionName_United States' ,'Occupation_Professional','StateProvinceName_California','City_London', 'Gender_F','MaritalStatus_S','Occupation_Skilled Manual','Education_Partial College','Occupation_Clerical','Education_High School','Education_Graduate Degree','Occupation_Manual']

y = train_dataset.BikeBuyer
X = data_features[features_select]
'NumberChildrenAtHome'               
,'MaritalStatus_M'                    
,'Gender_M'                           
,'CountryRegionName_United States'    
,'Occupation_Professional'            
,'Education_Bachelors'                
,'StateProvinceName_California'       
,'City_London' 
,'Gender_F'                           
,'MaritalStatus_S'                    
,'Occupation_Skilled Manual'          
,'Education_Partial College'          
,'Occupation_Clerical'                
,'Education_High School'              
,'Education_Graduate Degree'          
,'Occupation_Manual'

Xtest = test_data_features[features_select]

#validating our model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


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



predictions = model.predict(Xtest)

print(mean_absolute_error(y_test, val_predictions))
print (predictions)

my_submission = pd.DataFrame({'CustomerID':test_dataset.CustomerID,'BikeBuyer':predictions.astype(int)}).set_index('CustomerID')

my_submission2 = pd.DataFrame({'CustomerID':test_dataset.CustomerID,'BikeBuyer':predictions.astype(int)})

print(my_submission)
print(my_submission2)

my_submission.to_csv('7th_submission.csv', index=False)
my_submission2.to_csv('seventh_submission.csv', index=False)
















