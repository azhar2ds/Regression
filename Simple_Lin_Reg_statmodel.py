import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

d=pd.read_csv('https://raw.githubusercontent.com/llSourcell/linear_regression_demo/master/challenge_dataset.txt', header=None)

d.columns=['First','Second']

#read data

x_values = pd.DataFrame(d['First'].values)
y_values = pd.DataFrame(d['Second'].values)
#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

print('Coefficient:',body_reg.coef_)
print('Intercept:',body_reg.intercept_)
print('Score:',body_reg.score(x_values,y_values))


#similarly find regression using stats_models.api

import statsmodels.api as sm

s=sm.OLS(y_values,x_values).fit()
#prediction=s.prediction(x_values)
print('params Value:',s.params)
print(s.summary())

# Similarly find regressing using formula api in statsmodel

import statsmodels.formula.api as smf
# formula: response ~ predictors
est = smf.ols(formula='Second ~ First', data=d).fit()
print('\n\n\nRsquared Value:',est.rsquared)
print(est.summary())