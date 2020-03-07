
from statsmodels.api import OLS
import statsmodels.api as sm
from sklearn import datasets
import pandas as pd


data = datasets.load_boston()

# Set the features  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Set the target
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df["RM"]
y = target["MEDV"]


model = sm.OLS(y, X).fit()


print(model.summary())
print(45*'||')
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())
print(45*'||')
X = df[["RM", "LSTAT"]]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())