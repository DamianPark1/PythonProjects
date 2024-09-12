#Import libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Import dataset
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

#Print out first 5 rows to investigate Dataframe Structure, columns, values. 
print(df.head())

#Calculate the mean total production of honey per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
#print(prod_per_year)

#Create X variable for the Years in the Dataset
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
#Create y variable for the mean total honey production that year
y = prod_per_year['totalprod']

#Plot data in a scatterplot to assess linear relationships
plt.scatter(X, y)

#Create Linear regression model from scikit-learn 
regr = linear_model.LinearRegression()
#Fit model to the data using .fit()
regr.fit(X, y)
#print out Slope using .coef_ and intercept using .intercept_
#print(regr.coef_[0])
#print(regr.intercept_)

#Create list for predictions of LinReg model on the X data
y_predict = regr.predict(X)
#Plot LinReg prediction line over scatterplot
plt.title("Mean honey production per year")
plt.xlabel('Year')
plt.ylabel('Mean Honey Production')
plt.plot(X, y_predict)
plt.show()
plt.clf()


#Create future dataset array for future LinReg predictions of honey production
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

#Create future prediction LinReg Line
future_predict = regr.predict(X_future)
#Plot future prediction line on new plot
plt.title("Future Predictions of honey production per year")
plt.xlabel('Year')
plt.ylabel('Mean Honey Production')
plt.plot(X_future, future_predict)
plt.show()
plt.clf()
