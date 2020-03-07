from sklearn.datasets import load_boston
import pandas as pd
from sklearn import linear_model
import seaborn as sns

#importing builtin dataset for Boston House Prediction
d=load_boston()
X=pd.DataFrame(d.data, columns=d.feature_names)
y=d.target

print(len(dir(linear_model)))
lm=linear_model.LinearRegression()
lm.fit(X,y)
p=lm.predict(X)
print(p[:5])
print(y[:5])
print('Coefficients:',lm.coef_)
print('Y-Intercept:',lm.intercept_)
print('Score:',lm.score(X,y)*100)
print(X.columns)
sns.pairplot(X, x_vars=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'], y_vars=y, height=7, aspect=0.7, kind='reg')
