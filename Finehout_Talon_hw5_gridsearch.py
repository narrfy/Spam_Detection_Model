from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np



df = pd.read_csv("reduced_rotten_tomatoes_movie_reviews.csv")
documents = df["reviewText"][0:500]
true_sentiment = df['scoreSentiment'][0:500]
#print("True Sentiment",true_sentiment)

bow = CountVectorizer()
x = bow.fit_transform(documents)
label_encoder = LabelEncoder()
encoded = label_encoder.fit_transform(true_sentiment)

dt = DecisionTreeClassifier()
lr = LogisticRegression()

parameters_dt = {'criterion':["gini","entropy"],'max_depth':[10,15,20]}
parameters_lr = {'penalty':["l1","l2"],'solver':['liblinear']}
cv_dt = GridSearchCV(dt,parameters_dt,verbose=2)
cv_dt.fit(x,encoded)
cv_lr = GridSearchCV(lr,parameters_lr,verbose=2)
cv_lr.fit(x,encoded)

print(cv_dt.best_estimator_)
print(cv_lr.best_params_)




