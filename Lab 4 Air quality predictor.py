import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(r"D:\Dataset\city_day.csv")

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')  # Specify the correct date format here
data['Year'] = data['Date'].dt.year

# Selecting relevant features and target variable
X = data[['Year', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].copy()
y = data['AQI'].copy()

# Handling missing values
data.dropna(subset=['AQI'], inplace=True)
X = X.loc[data.index]  
y = y.loc[data.index]  

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print("Training Mean Squared Error:", mse_train)

y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Testing Mean Squared Error:", mse_test)

# Plotting actual vs. predicted AQI for test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs. Predicted AQI (Test Set)")
plt.grid(True)
plt.show()

# Predicting for user input city over the years
city = input("Enter the city: ")
predicted_aqi_all_years = {}
for year in data['Year'].unique():
    filtered_data = data[(data['City'] == city) & (data['Year'] == year)]
    if not filtered_data.empty:
        X_input = filtered_data[['Year', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
        predicted_aqi = model.predict(X_input)
        predicted_aqi_all_years[year] = predicted_aqi.mean()

# Plotting predicted AQI over the years
plt.figure(figsize=(10, 6))
plt.plot(list(predicted_aqi_all_years.keys()), list(predicted_aqi_all_years.values()), marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Predicted AQI")
plt.title("Predicted AQI over the Years for " + city)
plt.grid(True)
plt.show()

# Plotting actual AQI over the years
actual_aqi_all_years = data[data['City'] == city].groupby('Year')['AQI'].mean()
plt.figure(figsize=(10, 6))
plt.plot(actual_aqi_all_years.index, actual_aqi_all_years.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Actual AQI")
plt.title("Actual AQI over the Years for " + city)
plt.grid(True)
plt.show()
