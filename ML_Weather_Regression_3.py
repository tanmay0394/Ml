import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset (replace 'path_to_dataset' with the actual path)
url = r"weatherHistory.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Select relevant columns (humidity and apparent temperature)
data = df[['Humidity', 'Apparent Temperature (C)']]

# Drop rows with missing values
data = data.dropna()

# Split the dataset into features (X) and target variable (y)
X = data[['Humidity']]
y = data['Apparent Temperature (C)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the performance of the regression model
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-Square: {r2:.4f}")

# Visualize the simple linear regression model
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.title('Simple Linear Regression Model')
plt.legend()
plt.show()

y = a + bx
b= (n∑xy) − (∑x)(∑y) / (n∑x2) − (∑x)2
a= ∑y − b(∑x) / n
          


#Theory :-


# 1. What is Simple Linear Regression?
# 
# **Answer:**
# Simple Linear Regression is a statistical method that allows us to summarize and study the relationship between two continuous variables. It assumes that there is a linear relationship between the independent variable (predictor) and the dependent variable (response). The method aims to find the best-fitting line through the data points that minimizes the sum of squared differences between the observed and predicted values.
# 
# 2. Why is Simple Linear Regression Used?
# 
# **Answer:**
# Simple Linear Regression is used to model the relationship between two variables when it is believed that one variable (independent) influences the other (dependent) in a linear way. It is particularly useful for predicting the value of the dependent variable based on the value of the independent variable.
# 
# 
# 3. **What is the Purpose of Splitting the Data into Training and Testing Sets?**
# 
# **Answer:**
# The purpose of splitting the data into training and testing sets is to assess the performance of the model on new, unseen data. The model is trained on the training set and then evaluated on the testing set to estimate how well it will generalize to new, unseen data. This helps to avoid overfitting, where the model performs well on the training data but fails to generalize to new data.
# 
# 4. **What are Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-Squared (R2) Metrics?**
# 
# **Answer:**
# - Mean Squared Error (MSE): It measures the average squared difference between the actual and predicted values. A lower MSE indicates better model performance.
#     
#           n
# MSE = 1/n ∑   (yi - y^)^2
#           i=1
#   
# - **Mean Absolute Error (MAE):** It measures the average absolute difference between the actual and predicted values. Like MSE, a lower MAE indicates better model performance.
# 
#     
#           n
# MAE = 1/n ∑   |yi - y^|
#           i=1
# 
# 
# - **R-Squared (R2):** It represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). R2 values range from 0 to 1, with 1 indicating a perfect fit.
#         
#           ∑ (yi - y^)^2
# R^2 = 1 - ___________
#           ∑ (yi - y-)^2
#           
#                                  ________
# (Root Mean Square Error RMSE) = / MSE
# 
# 
# 5. **Explain the Steps Involved in Visualizing a Simple Linear Regression Model.**
# 
# Answer:
# 1. Scatter Plot:** Plot the scatter plot of the independent variable (e.g., Humidity) against the dependent variable (e.g., Apparent Temperature) to visualize the data distribution.
# 
# 2. **Regression Line:** Plot the regression line based on the trained model's coefficients. The line represents the best fit through the data points.
# 
# 3. **Labels and Title:** Add labels to the x-axis and y-axis for clarity. Include a title that describes the purpose of the plot, such as "Simple Linear Regression Model."
# 
# 4. **Show the Plot:** Display the plot to visualize how well the regression line fits the data points.
