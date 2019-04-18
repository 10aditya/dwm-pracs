import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("Y test:")
print(y_test)
print()

# Fitting Decision Tree Classification to the Training set using entrory
classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1)

print("Results using Entropy Criteria")
print()
print("Y prediction:")
print(y_pred1)
print()
print("Confusion Matrix: ")
print(cm1)

# Fitting Decision Tree Classification to the Training set using gini index
classifier2 = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(y_test, y_pred2)
print()
print("Results using Gini Index Criteria")
print()
print("Y prediction:")
print(y_pred2)
print()
print("Confusion Matrix: ")
print(cm2)