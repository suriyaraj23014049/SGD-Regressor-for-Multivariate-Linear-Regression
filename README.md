# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model Evaluation 
8.End
```
## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SURIYA RAJ K
RegisterNumber: 212223040216

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![Screenshot 2024-09-04 141320](https://github.com/user-attachments/assets/8883c91c-9315-4a4b-b022-fcb3d7caf3f5)
~~~
df.info()
~~~
![Screenshot 2024-09-04 141326](https://github.com/user-attachments/assets/68f1ff5e-688d-4192-90ed-de3ab70f1ffd)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![Screenshot 2024-09-04 141334](https://github.com/user-attachments/assets/48a3a046-c509-4aff-be9f-27acf064cbcd)
```
Y=df[['AveOccup','target']]
Y.info()
```
![Screenshot 2024-09-04 141342](https://github.com/user-attachments/assets/e926b7c4-dd05-4188-bbe7-fe69912303ac)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
![Screenshot 2024-09-04 141402](https://github.com/user-attachments/assets/6a80e029-dc8b-407e-9813-d92ad0d8015f)
```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
```
print(X_train)

![Screenshot 2024-09-04 141409](https://github.com/user-attachments/assets/989301ca-c7bf-40d0-a1d0-b2fe647597d3)

```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

```

## Output:
![Screenshot 2024-09-04 141416](https://github.com/user-attachments/assets/04a3ffe6-d120-4d58-8cab-f6c9cf6220f3)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
