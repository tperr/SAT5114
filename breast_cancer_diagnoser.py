from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

cancer = datasets.load_breast_cancer()
#print(cancer)

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) 
gnb = GaussianNB()
cv_scores = cross_val_score(gnb, x_train, y_train, cv=10)
grid = GridSearchCV(estimator=gnb, param_grid={"var_smoothing": np.logspace(0,-9, num=100)}, cv=10, n_jobs=1) # adding var_smoothing seems to make it less accurate unless you have a higher cv, but do not know what else to do
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
print("SD of accuracty:", cv_scores.std())
print("Best parameters:", grid.best_params_)

cm = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Sensitivity:", cm[1, 1]/(cm[1, 0] + cm[1, 1]))
print("Specificity:", cm[0, 0]/(cm[0, 0] + cm[0, 1]))
