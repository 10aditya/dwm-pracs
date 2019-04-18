import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

label_encoder = LabelEncoder()
x[:,3] = label_encoder.fit_transform(x[:,3])

one_hot_encoder = OneHotEncoder(categorical_features=[3])
x = one_hot_encoder.fit_transform(x).toarray()

# avoiding dummy variable trap
x = x[:,1:]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

pred = regressor.predict(x_test)

# print ( y_test, '\n',pred)

#backward elimination
x = np.append(arr=np.ones((50,1)).astype(int),values = x,axis=1)

x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_ols.summary(),'\n')

x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_ols.summary(),'\n')

x_opt = x[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_ols.summary(),'\n')

x_opt = x[:,[0,3]]
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_ols.summary(),'\n')

regressor_ols.predict()