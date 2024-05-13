import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"D:\Dataset\Housing.csv")

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Splitting the dataset into the features and the target variable
X = data.drop("price", axis=1)
y = data["price"]

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Model Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Function to predict price and visualize results
def predict_and_visualize(attributes):
    # Predicting the price
    predicted_price = model.predict(attributes)
    print("Predicted Price:", predicted_price[0])

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    # Scatter plot of actual vs predicted prices
    axes[0, 0].scatter(y_test, y_pred, color='blue')
    axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    axes[0, 0].set_xlabel("Actual Price")
    axes[0, 0].set_ylabel("Predicted Price")
    axes[0, 0].set_title("Actual vs Predicted Price")

    # Residual plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel("Predicted Price")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='orange')
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Histogram of Residuals")

    # Regression plot
    sns.regplot(x=y_test, y=y_pred, ax=axes[1, 1], color='purple')
    axes[1, 1].set_xlabel("Actual Price")
    axes[1, 1].set_ylabel("Predicted Price")
    axes[1, 1].set_title("Regression Plot")

    plt.tight_layout()
    plt.show()

# Take user input for attributes
attributes = []
for column in X.columns:
    value = input(f"Enter value for {column}: ")
    if value.isdigit():  # Check if the input is numeric
        attributes.append(float(value))
    elif value.lower() == 'yes':  # Convert categorical values to binary
        attributes.append(1)
    elif value.lower() == 'no':
        attributes.append(0)
    else:
        print(f"Invalid input for {column}")

# Convert the attributes to array and predict price
attributes = np.array(attributes).reshape(1, -1)
predict_and_visualize(attributes)