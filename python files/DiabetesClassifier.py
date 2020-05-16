# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:12:16 2020

@author: Hitesh Kumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib 


dataset = pd.read_csv("diabetes.csv")

dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 101)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#SVM

from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'rbf', random_state = 0)
classifierSVM.fit(X_train, y_train)
joblib.dump(classifierSVM, 'd_svm.pkl') 

#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
joblib.dump(classifierNB, 'd_nb.pkl') 



#RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 101)
classifierRF.fit(X_train, y_train)
joblib.dump(classifierRF, 'd_rf.pkl') 



#KNN

from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)
joblib.dump(classifierKNN, 'd_knn.pkl') 

#DECISION TREES

from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)
joblib.dump(classifierDT, 'd_dt.pkl') 


preg = int(input())
glucose = float(input())
bp = float(input())
st = float(input())
insulin = float(input())
bmi = float(input())
dpf = float(input())
age = int(input())

test = np.array([preg,glucose,bp,st,insulin,bmi,dpf,age])

test = test.reshape(1,-1)

test = sc.transform(test)
print(test)



y_predSVM = classifierSVM.predict(test)
y_predRF = classifierRF.predict(test)
y_predDT = classifierDT.predict(test) 
y_predLR = classifierLR.predict(test) 
y_predNB = classifierNB.predict(test) 
y_predKNN = classifierKNN.predict(test)

print(int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predNB),int(y_predKNN),int(y_predLR))

results = [int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predNB),int(y_predKNN),int(y_predLR)]

confidence = results.count(1)*100/len(results)

print(confidence)





