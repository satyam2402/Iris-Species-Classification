#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:52:00 2020

@author: harshvardhan
"""

#Importing the data set
import pandas as pd

irisdataset = pd.read_csv("Iris.csv")


#Preprocessing the data set

# 1. Since there is no need of id we can remove this
irisdataset = irisdataset.drop(columns = ["Id"], axis = 1)

# 2. Separating dependant and independant variable
x = irisdataset.iloc[:,:-1].values
y = irisdataset.iloc[:,-1].values

#Splitting the data into train and test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = .80)

#Using Random Forest Algorithm 
from sklearn.ensemble import RandomForestClassifier

#Noting the accuracy starting with 1 estimator till 50

estimators = []
accuracy = []

for n in range(1, 51):
    
    #Creating the classifier
    classifier = RandomForestClassifier(n_estimators = n, criterion = 'entropy', random_state = 0)

    #Fitting the data in the model
    classifier.fit(xtrain, ytrain)
    
    #Predicting the data from test set
    ypred = classifier.predict(xtest)
    
    #Fetching the accuracy of the model
    from sklearn.metrics import accuracy_score
    accuracyscore = accuracy_score(ytest, ypred)
    
    #Adding both the data in the list
    estimators.append(n)
    accuracy.append(accuracyscore)


#Plotting Accuracy vs Estimators
import matplotlib.pylab as plt
plt.bar(estimators, accuracy, width = 0.5)
plt.xlabel("Number of Estimator")
plt.ylabel("Accuracy Score")

#Printing max accuracy with which estimators
max_accuracy = max(accuracy)

#Max accuracy is with Estimator
estimatorwithmaxaccuracy = accuracy.index(max_accuracy)



