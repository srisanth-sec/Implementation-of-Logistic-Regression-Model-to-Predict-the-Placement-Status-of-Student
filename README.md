# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Load and preprocess the dataset (remove unwanted columns and convert categorical data).

2.Split the data into training and testing sets.

3.Train the Logistic Regression model using the training data.

4.Evaluate accuracy and visualize results using a sigmoid curve plot.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Srisanth R
RegisterNumber: 212225240156 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load & preprocess data
data = pd.read_csv("Placement_Data (1).csv").drop("salary", axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Plot (using one feature)
X1 = X.iloc[:, 0].values.reshape(-1, 1)
model_plot = LogisticRegression(max_iter=1000).fit(X1, y)

plt.scatter(X1, y)
x_vals = np.linspace(X1.min(), X1.max(), 100)
plt.plot(x_vals, model_plot.predict_proba(x_vals.reshape(-1,1))[:,1])

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()
```

## Output:
<img width="783" height="496" alt="Screenshot 2026-04-28 093355" src="https://github.com/user-attachments/assets/fc550543-c425-425a-8b75-f4a4cdacd9d1" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
