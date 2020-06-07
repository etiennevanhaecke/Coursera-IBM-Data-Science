# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:35:28 2020

@author: etienne.vanhaecke
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
#%matplotlib inline

file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)

df.head()

df.dtypes

df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

df.describe()

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#Use the method value_counts to count the number of houses with unique floor values, 
#use the method .to_frame() to convert it to a dataframe.
count_floor=df.floors.value_counts().to_frame()
type(count_floor)
count_floor

sns.boxplot(x="waterfront", y="price", data=df[["waterfront", "price"]])
sns.regplot(x="sqft_above", y="price", data=df[["sqft_above", "price"]])

df.corr()['price'].sort_values()

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

X = df[['sqft_living']]
Y = df['price']
lm2 = LinearRegression()
lm2.fit(X,Y)
lm2.score(X, Y)

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] 
X=df[features]
Y = df['price']
lm3 = LinearRegression()
lm3.fit(X,Y)
lm3.score(X, Y)

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

Pipeline
StandardScaler,PolynomialFeatures
LinearRegression

# criando o modelo usando pipeline
model = Pipeline(steps=Input)
# treinando o modelo
X=df[features]
Y = df['price']
model.fit(X, Y)
# avaliando o modelo
model.score(X, Y)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import Ridge
rdg = Ridge(alpha=0.1)
rdg.fit(x_train, y_train)
rdg.score(x_test, y_test)

rdg2 = Ridge(alpha=0.1)
poli=PolynomialFeatures(degree=2)
x_train_deg2 = poli.fit_transform(x_train)
x_test_deg2 = poli.fit_transform(x_test)
rdg2.fit(x_train_deg2, y_train)
rdg2.score(x_test_deg2, y_test)


