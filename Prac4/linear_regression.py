import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

exp = dataset.iloc[:,0:1].values
sal = dataset.iloc[:,1:2].values

exp_train,exp_test,sal_train,sal_test = train_test_split(exp,sal,test_size=1/4,random_state=0)

"""
scalar = StandardScaler(with_mean=True)
exp_train = scalar.fit_transform(exp_train)
exp_test = scalar.fit_transform(exp_test)
sal_train = scalar.fit_transform(sal_train)
sal_test = scalar.fit_transform(sal_test)
"""

regressor = LinearRegression()
regressor.fit(exp_train,sal_train)

predictions = regressor.predict(exp_test)

print(exp_test, '\n', sal_test, '\n', predictions)

plt.scatter(exp_train,sal_train, color='red')
plt.plot(exp_train,regressor.predict(exp_train), color='blue')
plt.title('Salary vs Experience (train data)')
plt.xlabel('years pf experience')
plt.ylabel('salary')
plt.show()

plt.scatter(exp_test,sal_test, color='red')
plt.plot(exp_train,regressor.predict(exp_train), color='blue')
plt.title('Salary vs Experience (test data)')
plt.xlabel('years pf experience')
plt.ylabel('salary')
plt.show()
