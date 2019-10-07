import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("ccdefault.csv")
df.head()

#in sample 
score =np.empty(shape=10)
y = df['DEFAULT']
X = df.drop('DEFAULT', axis=1)
for k in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y,random_state=k)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_train)
    score[k-1] = accuracy_score(y_train,y_pred)
    print("random state:",k,"score:",score[k-1])
print("mean of accuracy score:",np.mean(score),"std of accuracy score:",np.std(score))


#out of sample 
score =np.empty(shape=10)
y = df['DEFAULT']
X = df.drop('DEFAULT', axis=1)
for k in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y,random_state=k)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score[k-1] = accuracy_score(y_test,y_pred)
    print("random state:",k,"score:",score[k-1])
print("mean of accuracy score:",np.mean(score),"std of accuracy score:",np.std(score))

from sklearn.model_selection import cross_val_score

#cv
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
score = cross_val_score(clf,X_test,y_test,cv=10)
print("individual fold accuracy scores:",score)
print("mean:",np.mean(score),"std:",np.std(score))
 
