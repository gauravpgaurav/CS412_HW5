#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:35:56 2018

@author: gauravpant
"""
from dataApp import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

target = 'Empathy'
#target = 'Spending on healthy eating'

y_train = training[[target]].copy()
X_train_p = training.drop([target], axis=1)

y_dev = dev[[target]].copy()
X_dev_p = dev.drop([target], axis=1)

y_test = test[[target]].copy()
X_test_p = test.drop([target], axis=1)

X=X_train_p.iloc[:,:-1].values   #this will convert dataframe to object
X_train = pd.DataFrame(X)

X=X_dev_p.iloc[:,:-1].values   #this will convert dataframe to object
X_dev = pd.DataFrame(X)

X=X_test_p.iloc[:,:-1].values   #this will convert dataframe to object
X_test = pd.DataFrame(X)

# Create our imputer to replace missing values with the mean e.g.
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X_dev)
X_dev_imp = imp.transform(X_dev)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X_test)
X_test_imp = imp.transform(X_test)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(y_train)
y_train_imp = imp.transform(y_train)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(y_dev)
y_dev_imp = imp.transform(y_dev)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(y_test)
y_test_imp = imp.transform(y_test)

y_train_imp=y_train_imp.reshape(len(y_train_imp))
#Y= y_train_imp

clf = RandomForestClassifier(n_estimators=10)
resArray = np.zeros(10)
for i in range(0, 10):
    clf = clf.fit(X=X_train_imp, y=y_train_imp)
    result = clf.score(X=X_dev_imp, y=y_dev_imp) 
    resArray[i] = result

print(str(np.mean(resArray)))
#0.356435643564