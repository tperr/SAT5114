from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import random
import numpy as np

print("Loading dataset...")
leukemia = datasets.fetch_openml(data_id=1104)
print("Loaded dataset, splitting data into training/testing sections")
x_train, x_test, y_train, y_test = train_test_split(leukemia.data, leukemia.target, test_size=0.3) 
gnb = GaussianNB()
print("Splitted, setting up randomsearch...")
grid = RandomizedSearchCV(estimator=gnb, param_distributions={"var_smoothing": np.logspace(0,-9, num=100)}, cv=5, n_jobs=1, random_state=random.randint(0, 1000))
print("Fitting...")
grid.fit(x_train, y_train)
print("Done! Metrics:")
y_pred = grid.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Sensitivity:", cm[1, 1]/(cm[1, 0] + cm[1, 1]))
print("Specificity:", cm[0, 0]/(cm[0, 0] + cm[0, 1]))