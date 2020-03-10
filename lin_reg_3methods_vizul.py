import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


d=pd.read_csv('https://raw.githubusercontent.com/azhar2ds/DataSets/master/headbrain.csv')
X=pd.DataFrame(d['Head Size(cm^3)'])
Y=pd.DataFrame(d['Brain Weight(grams)'])

#Using LinearRegression from Sklearn.linear_model
l=LinearRegression()
l.fit(X,Y)
print('Score:',l.score(X,Y))
print('Coefficient:',l.coef_)
print('Intercept:',l.intercept_)
i=X[202:203].values
pre=l.predict(i)
print('Actual value is:',i,'. But Predicted value is:',pre)

#Visualization!!!!

sns.pairplot(d,x_vars=['Head Size(cm^3)'],y_vars=['Brain Weight(grams)'], height=6,aspect=1.5, kind='reg')



#We can also calculate the coefficients with numpy

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print("values of BETA 1 & BETA 0 using numpy:",b1, b0)
'''
plt.scatter(x, y)
plt.plot(x, l.predict(x))
plt.show()
'''