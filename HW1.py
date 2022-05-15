#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:01:53 2021

@author: yingxue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### import dataset
data=pd.read_csv('/home/yingxue/Desktop/auto-mpg.csv')
### drop several columns
data=data.drop(columns=['car name','origin','model year'])
### find the quantile of mpg column
mpg=data['mpg']
low=data['mpg'].quantile(0.33)
mid=data['mpg'].quantile(0.66)
### select the LOWmpg data set, MIDmpg dataset and the HIGHmpg dataset. 
LOWmpg=data.loc[data['mpg'] <= low]
LOWY=pd.DataFrame([1]*130)
MIDmpg=data.loc[(data['mpg'] > low) & (data['mpg'] <= mid)]
MIDY=pd.DataFrame([2]*130)
HIGHmpg=data.loc[data['mpg'] > mid]
HIGHY=pd.DataFrame([3]*132)

from sklearn.model_selection import train_test_split
###split training and test set for LOWmpg,MIDmpg,HIGHmpg
low_train,low_test,lowy_train,lowy_test=train_test_split(LOWmpg,LOWY,test_size=0.2)
mid_train,mid_test,midy_train,midy_test=train_test_split(MIDmpg,MIDY,test_size=0.2)
high_train,high_test,highy_train,highy_test=train_test_split(HIGHmpg,HIGHY,test_size=0.2)
### combine to get the global training set the global test set
Train = pd.concat([low_train,mid_train,high_train])
Ytrain= pd.concat([lowy_train,midy_train,highy_train])
Train=Train.drop(columns=['mpg'])
Test= pd.concat([low_test,mid_test,high_test])
Ytest= pd.concat([lowy_test,midy_test,highy_test])
Test=Test.drop(columns=['mpg'])

### KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
model.fit(Train,Ytrain)
#Predict Output
y_train_pred= model.predict(Train) 
y_test_pred=model.predict(Test)

from sklearn import metrics
# Model Accuracy
print("Accuracy of training set:",metrics.accuracy_score(Ytrain, y_train_pred))
print("Accuracy of test set:",metrics.accuracy_score(Ytest, y_test_pred))
## confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Ytrain, y_train_pred)
confusion_matrix(Ytest, y_test_pred)


### knn for k=3,5,7,9,11,13,15,17,19,29,39
import math
TraAcc=[]## store the accuracy on training set for different k in this list
TestAcc=[]## store the accuracy on test set for different k in this list
margin=[]## store the margin on test set in this list
for k in [3,5,7,9,11,13,15,17,19,29,39]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Train,Ytrain)
    y_train_pred= model.predict(Train) 
    y_test_pred=model.predict(Test)
    acc1=metrics.accuracy_score(Ytrain, y_train_pred)
    acc2=metrics.accuracy_score(Ytest, y_test_pred)
    margin.append(math.sqrt(acc2*(1-acc2)/79))
    TraAcc.append(acc1)
    TestAcc.append(acc2)


l=np.asarray([3,5,7,9,11,13,15,17,19,29,39])
###Compute the 90% confidence interval [CI1 CI2]
CI1=TestAcc-1.6*np.asarray(margin)
CI2=TestAcc+1.6*np.asarray(margin)

###plot the accuracy curve on training set(red), test set(black) and the 90% confidence 
###interval of accuracy on test set(yellow area) in the same figure
plt.fill_between(l,CI1,CI2,color='yellow',label='90% CI')
plt.plot(l,TraAcc,'*--',color='red',label='TrainAccuracy')
plt.plot(l,TestAcc,'.--',color='black',label='TestAccuracy')
plt.legend()












