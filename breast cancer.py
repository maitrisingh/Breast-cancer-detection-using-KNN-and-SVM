#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:45:34 2020

@author: code
"""

import sys
import matplotlib
import numpy
import pandas
import sklearn



print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id','clump_thickness','uniform_cell_size','uniform cell shape','marginal adhesion','single_epithelial_size','bare nuclei','bland_chromatin','normal_nucleoli','mitoses','class'
        ]
df = pd.read_csv(url,names=names)


#preprocess the data
df.replace('?',-99999,inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)

#print the shape of the dataset
print(df.shape)

# Do dataset visualizations

print(df.loc[698])
print(df.describe())

#print histogram for each variable
df.hist(figsize =(10,10))
plt.show()

#create scatter plot matrix
scatter_matrix(df, figsize =(18,18))
plt.show()

#create X and Y dataset for training
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2)

#specify testing options
seed = 8
scoring = 'accuracy'

#define the models 
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM' , SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10 , random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv= kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean() , cv_results.std())
    print(msg)  
    
    #make prediction on  validation dataset
    
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(name)
        print(accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))
        
        clf = SVC()
        
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(accuracy)
        
        example = np.array([[4,2,1,1,1,2,3,2,10]])
        example = example.reshape(len(example), -1)
        prediction = clf.predict(example)
        print(prediction)
        
