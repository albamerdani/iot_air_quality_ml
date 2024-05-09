import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Step 5: Predict future alarm values for the next hour
#last_hour_data = X.iloc[-1].values.reshape(1, -1)  # Use the last hour's data
# Use all historical data
predicted_alarm = model.predict(X)  #last_hour_data
print("Predicted alarm for the next hour based on all historical data:", predicted_alarm[-1])

# Step 6: Visualize the predictions
plt.figure(figsize=(10, 5))

# Plot actual alarms
plt.plot(np.array(df.index), df['alarm_status'].values, label='Actual Alarm', color='blue', marker='o')

# Plot predicted alarms
plt.plot(np.array(df.index), predicted_alarm, label='Predicted Alarm', color='red')

plt.title('Actual vs Predicted Alarm')
plt.xlabel('Time')
plt.ylabel('Alarm Status (1: On, 0: Off)')
plt.legend()
plt.show()


# Step 7: Get random sample from the dataset to use as input values
num_samples = 10  # Number of samples to take
random_samples = df.sample(num_samples)

# Step 8: Predict alarms for the selected samples
input_values = random_samples[['pm2_5', 'pm10', 'temp', 'trysnia', 'lageshtira', 'hour', 'day_of_week']].values
predicted_alarms = model.predict(input_values)

#input_value = df.sample(n=10)  # Randomly select 10 rows as input
#predicted_alarms = model.predict(input_value.drop(columns='alarm_status'))

# Step 9: Plot the predicted alarms based on the selected samples
plt.figure(figsize=(10, 8))

# Plot the input values
for i, input_value in enumerate(input_values):
    plt.plot(range(len(input_value)), input_value, label=f'Input {i+1}')


plt.plot(np.array(df.index), df['alarm_status'].values, label='Actual Alarm', color='blue', marker='o')
# Plot the predicted alarms
#plt.scatter([len(input_value)], predicted_alarms[:-1], color='red', marker='o', label='Predicted Alarm')
plt.scatter(np.array(input_value), predicted_alarms[:-1], color='red', marker='o', label='Predicted Alarm')

plt.title('Predicted Alarm Based on Other Inputs')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value / Predicted Alarm (1: On, 0: Off)')
plt.legend()
plt.grid(True)
plt.show()

