# Required Packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


#Converting built in dataset in Pandas DataFrame & DataSeries
d=datasets.fetch_california_housing()
header=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
df=pd.DataFrame(d.data, columns=header)

#Adding the new target column to a DataFrame
df = df.assign(Price=d.target) 

X=df[df.columns[:-1]]
y=df[df.columns[-1]].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
r=linear_model.LinearRegression()
r.fit(X_train,y_train)
pre=r.predict(X_test)

i=4544
print("\n\nPredicted Value:",pre[i])
print("Actual Value:",y_test[i])
print('Worst Prediction ever!!!')

print("Y-Intecept:",r.intercept_)
print("\nCoefficient values:\n",r.coef_)