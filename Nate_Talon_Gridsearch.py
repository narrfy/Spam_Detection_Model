from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import numpy as np


df = pd.read_csv("train.csv")
documents = df["v2"][0:5000]
true_sentiment = df['v1'][0:5000]

bow = CountVectorizer()
label_encoder = LabelEncoder()

x = bow.fit_transform(documents)
encoded = label_encoder.fit_transform(true_sentiment)

dt = DecisionTreeClassifier()
svm = SVC()
rfc = RandomForestClassifier()

parameters_dt = {'criterion':["gini","entropy"],'max_depth':[10,15,20,25,30,35,40]}
parameters_svm = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}
parameters_rfc = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}

cv_dt = GridSearchCV(dt, parameters_dt, verbose=2)
cv_dt.fit(x, encoded)

cv_svm = GridSearchCV(svm, parameters_svm, verbose=2)
cv_svm.fit(x, encoded)

cv_rfc = GridSearchCV(rfc, parameters_rfc, verbose=2)
cv_rfc.fit(x, encoded)


print("Best parameters for Decision Tree Classifier:")
print(cv_dt.best_params_)  # Best params for DT
print()

print("Best parameters for Support Vector Machine Classifier:")
print(cv_svm.best_params_) # Best params for SVM
print()

print("Best parameters for Random Forest Classifier:")
print(cv_rfc.best_params_) # Best params for RFC

'''
Best parameters for Decision Tree Classifier:
{'criterion': 'gini', 'max_depth': 30}

Best parameters for Support Vector Machine Classifier:
{'C': 0.1, 'kernel': 'linear'}

Best parameters for Random Forest Classifier:
{'max_depth': 30, 'n_estimators': 100}
'''




