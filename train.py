import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import RandomForestRegressor as model 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler as scl
from sklearn.impute import SimpleImputer as imp

from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error,mean_squared_error
import csv

  
df = pd.read_csv("CleanedT.csv")
df2=df

np.random.seed(42)

pd.set_option('display.max_rows',35000)
pd.set_option('display.max_columns',35)
x= df2.drop(['target'], axis =1 )
y= df2['target']

def more(x):
    return (x-)

xtrain,xtest,ytrain,ytest=tts(x, y, test_size=0.25, random_state=150 )



scaler = scl()
scaler.fit(xtrain)

xtrain=scaler.transform(xtrain)
xtest=scaler.transform(xtest)

imputer=imp()
xtrain=imputer.fit_transform(xtrain)
xtest=imputer.transform(xtest)


clf=model( bootstrap= False,
 max_depth= 150,
 max_features=12,
 min_samples_leaf = 2,
 min_samples_split= 2,
 n_estimators= 2000)

#print(xtest.head())

clf.fit(xtrain,ytrain)
pred =clf.predict(xtest)
mae =mean_absolute_error(ytest,pred)
mse =mean_squared_error(ytest,pred)
nf =pd.DataFrame(data={"actual values":ytest,"predicted values":pred})
nf['differences']= nf['actual values']- nf['predicted values']



mape = np.mean(np.abs((ytest-pred)/ytest))*100


print('')
print(f'the f1 test score {clf.score(xtest,ytest)}')
print('')
print(f'the f1 train score {clf.score(xtrain,ytrain)}')
print('')
print(f"the MAE value is {mae}")
print('')
print(f"the MSE value is {mse}")
print('')
print(f"the MAPE value is {mape}")
print('')

clear = open ("compare.csv","w")
clear.truncate()
clear.close
nf.to_csv('compare.csv',mode='a', index=False)
