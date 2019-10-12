import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

y = df['DEFAULT']
X = df.drop('DEFAULT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y,random_state=42)

# Import RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#cv
rf = RandomForestClassifier(random_state=2)
rf.fit(X_train,y_train)
scores = cross_val_score(rf, X_train, y_train, cv=10)
print(scores)


#n_estimators
# Instantiate rf
for k in range(1,26):
    rf = RandomForestClassifier(n_estimators=k,
            random_state=2)
              
    rf.fit(X_train, y_train) 
    print("n estimators",k,"score:",rf.score(X_train, y_train))
    
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values( )

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh',color
='lightgreen')
plt.title('Features Importances')
plt.show()

print("My name is {Kainan Wang}")
print("My NetID is: {kainanw2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
