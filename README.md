# Ex03 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
  1. Import required libraries in python for Gradient Design.
  2. Upload the dataset and check any null value using .isnull() function.
  3. Declare the default values for linear regression.
  4. Calculate the loss usinng Mean Square Error.
  5. Predict the value of y.
  6. Plot the graph respect to hours and scores using scatter plot function.
   
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JANARTHANAN V K 
RegisterNumber: 212222230051
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate = 0.01, num_iters = 100):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta

data = pd.read_csv("..\Ex03\50_Startups.csv")
data.head()

X = data.iloc[1:, :-2].values
print(X)

X1 = X.astype(float)
scaler = StandardScaler()
y = data.iloc[1:,-1].values.reshape(-1,1)
print(y)

X1_scaled = scaler.fit_transform(X1)
Y1_scaled = scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)

theta = linear_regression(X1_scaled,Y1_scaled)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_scaled),theta).reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print("Prediction value:",pre)
```
## Output:

#### Data Info:
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/707d5cd8-6b43-4502-86be-371ff81d0520" width=40%>

#### X and Y Values:
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/4464d4e7-81bc-4e22-af54-53d17d047038" width=25%>
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/5760c007-55a2-40d2-971e-c2d97e35a40d">


#### X and Y Scaled:
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/7c130ed4-f795-40a5-9b3c-f655e455def5" width=35%>
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/608f89a5-6034-4bc2-8e18-0c4929ef2809">

#### Predicted Value:
<img src="https://github.com/Janarthanan2/ML_EX03_Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393515/50426660-f62a-4864-9ad6-499c8012f489">

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
