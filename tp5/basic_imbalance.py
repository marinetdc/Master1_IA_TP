#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:51:49 2020

@author: cecile
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

X,Y = make_classification(n_samples=150, n_features=20, n_informative=10, 
                                n_redundant=5, n_repeated=5, n_classes=2, 
                                n_clusters_per_class=2, weights=[0.9, 0.1], 
                                flip_y=0.01, class_sep=1.5, hypercube=True)
cnames=["M","m"]

X_app,X_test,Y_app,Y_test=train_test_split(X,Y,test_size=0.30,random_state=12)

print('*********** Rapport avec déséquilibre ************')
clf_nb = DummyClassifier(strategy="stratified")
clf_nb.fit(X_app,Y_app)
y_pred_nb = clf_nb.predict(X_test)
print('Maj:',classification_report(Y_test, y_pred_nb, target_names=cnames))

clf_nb = GaussianNB()
clf_nb.fit(X_app,Y_app)
y_pred_nb = clf_nb.predict(X_test)
print('NB: ',classification_report(Y_test, y_pred_nb, target_names=cnames))

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_app,Y_app)
y_pred_nb = clf_dt.predict(X_test)
print('DT: ',classification_report(Y_test, y_pred_nb, target_names=cnames))

clf_kppv = KNeighborsClassifier()
clf_kppv.fit(X_app,Y_app)
y_pred_nb = clf_kppv.predict(X_test)
print('KP: ',classification_report(Y_test, y_pred_nb, target_names=cnames))
