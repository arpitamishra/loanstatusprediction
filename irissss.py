import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 1)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
predicted = clf.predict(X_test)
print(predicted)







