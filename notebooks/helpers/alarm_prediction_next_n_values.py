import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

path_file = os.path.join("../../dataset", "matje.csv")

# Step 1: Load and preprocess the data
df = pd.read_csv(path_file)
df['created_at'] = pd.to_datetime(df['created_at'])  # Convert to datetime
df.set_index('created_at', inplace=True)  # Set timestamp as index
df.dropna(inplace=True)  # Drop rows with missing values

# Step 2: Feature engineering
# You can extract features like hour of the day, day of the week, etc.
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Step 3: Split the data (No need for splitting as we'll use all data)
X = df[['pm2_5', 'pm10', 'temp', 'trysnia', 'lageshtira', 'hour', 'day_of_week']]
y = df['alarm_status']

# Step 4: Train a machine learning model
# Define feature names explicitly
feature_names = X.columns.tolist()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 5: Predict next n alarm values based on all historic data
# Use all available data for prediction
last_data_point = X

future_predictions = model.predict(last_data_point)

print("Predicted alarm values for the next 50 records:", future_predictions[:50])

# Step 6: Visualize the predictions
plt.figure(figsize=(10, 6))

# Plot actual alarms
plt.plot(df.index.tolist(), df['alarm_status'].values, label='Actual Alarm', color='blue', marker='o')

# Plot predicted alarms
plt.plot(df.index.tolist(), [None]*(len(df.index)-len(future_predictions)) + list(future_predictions), label='Predicted Alarm', color='red')

plt.title('Actual vs Predicted Alarm')
plt.xlabel('Time')
plt.ylabel('Alarm Status (1: On, 0: Off)')
plt.legend()
plt.show()