# 2025.03.10 커밋 필요
# 프로젝트2 븟꽃분류기 만들기
# 이용희교수님과 열심히 만들어보자

from fileinput import filename

import joblib
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

iris_df = pd.read_csv('iris.csv')
# print(iris_df)
# print(iris_df['sepal_length'])
y = iris_df['species']
X = iris_df.drop('species', axis = 1)

kn = KNeighborsClassifier()
rfc = RandomForestClassifier()
model_kn = kn.fit(X, y)
model_rfc = rfc.fit(X, y)

joblib.dump(model_rfc, 'model_rfc.pkl')

# X_new = np.array([[3, 3, 3, 3]])
# kn ['versicolor'] [[0. 0.8 0.2]]
X_new = np.array([[5.0, 3.4, 3.5, 0.2]])
# kn ['setosa'] # [[1. 0. 0]]
# rfc ['setosa'] # [[1. 0. 0]]
# rfc ['setosa'] [[0.53 0.21 0.26]]
#X_new = np.array([[1, 4.2, 1.4, 7]])
# prediction = model_kn.predict(X_new)
prediction = model_rfc.predict(X_new)
print(prediction)
# probability = model_kn.predict_proba(X_new)
probability = model_rfc.predict_proba(X_new)
print(probability)
# print(y)
# print(X)