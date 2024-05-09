import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load the dataset
path_file = os.path.join("../../dataset", "matje.csv")
df = pd.read_csv(path_file)

# Convert the 'created_at' column to datetime and extract relevant time features
df['created_at'] = pd.to_datetime(df['created_at'])
df['hour'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek

# Select features for regression; exclude 'pm10' if predicting 'pm2_5' and vice versa
features = ['temp', 'trysnia', 'lageshtira', 'hour', 'day_of_week']
X = df[features]
y = df['pm2_5']  # Change this to 'pm10' if predicting 'pm10'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the regressor
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Visualize the actual vs predicted values
plt.figure(figsize=(14, 7))

# Scatter plot for actual vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual pm2_5 levels')
plt.ylabel('Predicted pm2_5 levels')
plt.title('Actual vs Predicted pm2_5 Levels')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line

# Line plot for actual vs predicted values over the test dataset
plt.subplot(1, 2, 2)
plt.plot(y_test.reset_index(drop=True), label='Actual pm2_5 levels')
plt.plot(pd.Series(y_pred, index=y_test.index).reset_index(drop=True), label='Predicted pm2_5 levels', alpha=0.7)
plt.xlabel('Test sample index')
plt.ylabel('pm2_5 level')
plt.title('Line Plot of Actual vs Predicted pm2_5 Levels')
plt.legend()

plt.tight_layout()
plt.show()
