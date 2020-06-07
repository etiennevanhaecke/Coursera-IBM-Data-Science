# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:37:53 2020

@author: etienne.vanhaecke
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline

URL="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv"
df = pd.read_csv(URL)
df.head()
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

df['loan_status'].value_counts()
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
X[0:5]
y = df['loan_status'].values
y[0:5]
standScal = preprocessing.StandardScaler().fit(X)
X= standScal.transform(X)
X[0:5]

#K Nearest Neighbor(KNN)
#Separation of the training data set betweeen training data set (80%) and the validation data set (20%)
from sklearn.model_selection import train_test_split
#Split of the training data set to separate 20% for the validation of the better k parameter
XTrain, XVal, yTrain, yVal = train_test_split(X, y, test_size=0.2)
print("len of validation data set: "+str(len(yVal)))
XVal.shape 
print("len of training data set: "+str(len(yTrain)))
XTrain.shape 

from sklearn.neighbors import KNeighborsClassifier
#Test of one model with k to 4
k = 4
model = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
model.fit(XTrain, yTrain)
yValPred = model.predict(XVal)
model.score(XVal, yVal)
#Definition of the best k changing k from 1 to 20
kMax = 20
#Array to conserve the accuracy for each k model
kScore=np.zeros(kMax)
#Loop sobre each k value
for k in range(1, kMax+1):
    modelK = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
    modelK.fit(XTrain, yTrain)
    kScore[k-1]=modelK.score(XVal, yVal)
    print('Valid Mean Accuracy Model %d-Nearest: %f' %(k, kScore[k-1]))
#Plot of the Valid Mean Accuracy in function of the k-nearest
plt.plot(range(1, kMax+1), kScore, 'go--')
#Selection of the k-nearest with higher mean accuracy
kVal=kScore.argmax()+1
accKNearMod = kScore[kVal-1]
print('The k-nearest with the higher mean accuracy %f is %d' %(accKNearMod, kVal))
modelKNN = KNeighborsClassifier(n_neighbors=kVal, weights='uniform', algorithm='auto')
modelKNN.fit(XTrain, yTrain)

    
#Tree Decision
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy', splitter='best', 
                             min_samples_split=2, min_samples_leaf=1)
model.fit(XTrain, yTrain)
model.score(XVal, yVal)
#Definition of the best min_samples_split and min_samples_leaf changing these from 1 to 20
splitMax = 20
leafMax = 20
#Array to conserve the accuracy for each combination of min_samples_split and min_samples_leaf
splitLeafScore=np.zeros((splitMax, leafMax))
for s in range(2, splitMax+1):
    for l in range(1, leafMax+1):
        modelSplitLeaf=DecisionTreeClassifier(criterion='entropy', splitter='best', 
                             min_samples_split=s, min_samples_leaf=l)
        modelSplitLeaf.fit(XTrain, yTrain)
        splitLeafScore[s-1, l-1]=modelSplitLeaf.score(XVal, yVal)
        print('Valid Mean Accuracy Model with %d split and %d leaf: %f' %(s, l, splitLeafScore[s-1, l-1]))
#Selection of the min samples split and min samples leaf parameters with higher mean accuracy
ind = np.unravel_index(splitLeafScore.argmax(axis=None), splitLeafScore.shape)
splitVal=ind[0]+1
leafVal=ind[1]+1
accTreeMod=splitLeafScore[splitVal-1, leafVal-1]
print('The Tree Model with the higher mean accuracy %f has %d min samples split and %d min samples leaf' 
      %(accTreeMod, splitVal, leafVal))
modelTree=DecisionTreeClassifier(criterion='entropy', splitter='best', 
                             min_samples_split=splitVal, min_samples_leaf=leafVal)
modelTree.fit(XTrain, yTrain)

#SVM
from sklearn import svm
model = svm.SVC(C=1, kernel='rbf', gamma='auto', max_iter=-1)
model.fit(XTrain, yTrain)
model.score(XVal, yVal)
#Definition of the best C penality parameter from 0.1 to 5
cMax = 50 #multiplicado por 10 para facilitar o loop
#Array to conserve the accuracy for each C penality value
cScore=np.zeros(cMax)
#Loop sobre each c value
for c in range(1, cMax+1):
    modelC = svm.SVC(C=c/10, kernel='rbf', gamma='auto', max_iter=-1)
    modelC.fit(XTrain, yTrain)
    cScore[c-1]=modelC.score(XVal, yVal)
    print('Valid Mean Accuracy Model with C Penality %f is %f' %(c/10, cScore[c-1]))
#Plot of the Valid Mean Accuracy in function of the C penality
plt.plot(range(1, cMax+1), cScore, 'go--')
#Selection of the C penality parameter with higher mean accuracy
cVal=(cScore.argmax()+1)/10
accSVCMod = cScore[int(cVal*10-1)]
print('The C penality parameter with the higher mean accuracy %f is %f' %(accSVCMod, cVal))
modelSVM = svm.SVC(C=cVal, kernel='rbf', gamma='auto', max_iter=-1)
modelSVM.fit(XTrain, yTrain)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=0.1, solver='liblinear',max_iter=10000)
model.fit(XTrain, yTrain)
model.score(XVal, yVal)
#Definition of the best C penality parameter from 0.01 to 5
cMax = 500 #multiplicado por 100 para facilitar o loop
#Array to conserve the accuracy for each C penality value
cScore=np.zeros(cMax)
#Loop sobre each c value
for c in range(1, cMax+1):
    modelC2=LogisticRegression(C=c/100, solver='liblinear',max_iter=10000)
    modelC2.fit(XTrain, yTrain)
    cScore[c-1]=modelC2.score(XVal, yVal)
    print('Valid Mean Accuracy Model with C Penality %f is %f' %(c/100, cScore[c-1]))
#Plot of the Valid Mean Accuracy in function of the k-nearest
plt.plot(range(1, cMax+1), cScore, 'go--')
#Selection of the C penality parameter with higher mean accuracy
cVal=(cScore.argmax()+1)/100
accLRMod = cScore[int(cVal*100-1)]
print('The C penality parameter with the higher mean accuracy %f is %f' %(accLRMod, cVal))
modelLR=LogisticRegression(C=cVal, solver='liblinear',max_iter=10000)
modelLR.fit(XTrain, yTrain)


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
URL = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv'
test_df = pd.read_csv(URL)
test_df.head()

#Preparation of the test data set like it has been done for the training data set
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df.head()
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()

test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()
test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
test_df[['Principal','terms','age','Gender','education']].head()

Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

XTest = Feature
XTest[0:5]
yTest = test_df['loan_status'].values
yTest[0:5]
XTest= standScal.transform(XTest)
XTest[0:5]

#Evaluation of each model with the best hiperparameters defined during the training
#KNN Model
accJacKNN=jaccard_similarity_score(y_true=yTest, y_pred=modelKNN.predict(XTest))
accF1KNN=f1_score(y_true=yTest, y_pred=modelKNN.predict(XTest), average='weighted')

#Tree Model
accJacTree=jaccard_similarity_score(y_true=yTest, y_pred=modelTree.predict(XTest))
accF1Tree=f1_score(y_true=yTest, y_pred=modelTree.predict(XTest), average='weighted')

#SVM
accJacSVM=jaccard_similarity_score(y_true=yTest, y_pred=modelSVM.predict(XTest))
accF1SVM=f1_score(y_true=yTest, y_pred=modelSVM.predict(XTest), average='weighted')

#LR
accJacLR=jaccard_similarity_score(y_true=yTest, y_pred=modelLR.predict(XTest))
accF1LR=f1_score(y_true=yTest, y_pred=modelLR.predict(XTest), average='weighted')

yhat_probab = modelLR.predict_proba(XTest)
accLossLR = log_loss(y_true=yTest, y_pred=yhat_probab)

#Reporting
report = pd.DataFrame()
report["Algorithm"]=['KNN', 'Decision Tree', 'SVM', 'LogisticRegression']
report["Jaccard"]=[accJacKNN, accJacTree, accJacSVM, accJacLR]
report["F1-score"]=[accF1KNN, accF1Tree, accF1SVM, accF1LR]
report["LogLoss"]=[np.NaN, np.NaN, np.NaN, accLossLR]
report