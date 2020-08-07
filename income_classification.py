# importing the dataset 
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
  
  
df = pd.read_csv('https://raw.githubusercontent.com/azhar2ds/DataSets/master/adult.csv')     

df = df.drop(['fnlwgt', 'educational-num'], axis = 1) 

category_col =['workclass', 'race', 'education', 'marital-status', 'occupation', 
               'relationship', 'gender', 'native-country', 'income']  
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 

for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping
    
X = df.values[:, 0:12] 
Y = df.values[:, 12] 

X_train, X_test, y_train, y_test = train_test_split( 
           X, Y, test_size = 0.3, random_state = 100) 
  
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                     random_state = 100, 
                                     max_depth = 5, 
                                     min_samples_leaf = 5) 
  
dt_clf_gini.fit(X_train, y_train) 
y_pred_gini = dt_clf_gini.predict(X_test) 
  
print("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test, y_pred_gini)*100 )



